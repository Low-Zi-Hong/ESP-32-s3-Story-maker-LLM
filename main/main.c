/*
 * esp32_storyteller.ino — on-device LLM inference for ESP32-S3
 * ------------------------------------------------------------
 * Runs a quantized Stories15M (Llama-2 architecture) entirely on-chip:
 *
 *   Tactic #1: weights memory-mapped from a raw flash partition (zero RAM)
 *   Tactic #2: INT4 (or INT8) weights, INT8 activations, integer matmuls
 *   Tactic #3: hot loop isolated in llm_matmul_rows() — PIE SIMD drop-in point
 *   Tactic #4: matmul rows split across both LX7 cores
 *
 * SETUP
 *  1. On your PC:  python3 export_model.py stories15M.bin tokenizer.bin \
 *                       model_esp.bin --bits 4
 *  2. Put partitions.csv next to this sketch. In Arduino IDE select your
 *     ESP32-S3 board, enable PSRAM (OPI for the Waveshare 5"), Flash 16MB.
 *  3. Upload the sketch, then flash the model image:
 *       esptool.py --chip esp32s3 write_flash 0x1F0000 model_esp.bin
 *  4. Serial Monitor @ 115200. Type a story opening, get a continuation.
 *     Commands:  /temp 0.8   /topp 0.9   /len 200   /stats
 *
 * Requires: ESP32-S3 with PSRAM (KV cache lives there) and 16MB flash.
 */

#include "esp_partition.h"
/* compat: Arduino-esp32 core 2.x (IDF4) used older names for the mmap API */
#include "esp_idf_version.h"
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
  #include "esp_spi_flash.h"
  typedef spi_flash_mmap_handle_t esp_partition_mmap_handle_t;
  #define ESP_PARTITION_MMAP_DATA SPI_FLASH_MMAP_DATA
#endif
#include "esp_heap_caps.h"
#include "llm_core.h"
#include "esp_log.h"
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_random.h"
#include "driver/gpio.h"

#include <sys/select.h>
#include <sys/time.h>

static const char *TAG = "TLM_CORE";

// ---------------- config ----------------
static float    g_temp   = 0.8f;
static float    g_topp   = 0.9f;
static int      g_maxlen = 180;     // tokens to generate per prompt

#include "driver/uart.h"

// 在 app_main 里初始化 UART0 接收（加在 load_model 之前）
static void uart_init() {
    uart_config_t cfg = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &cfg);
}

// ---------------- PSRAM-aware allocators (KV cache, logits) ----------------
static void* alloc_big(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) p = heap_caps_malloc(n, MALLOC_CAP_8BIT);   // fallback
  return p;
}
static void* alloc_small(size_t n) {                  // hot buffers -> SRAM
  void* p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) p = heap_caps_malloc(n, MALLOC_CAP_8BIT);
  return p;
}

// ---------------- tactic #4: dual-core matmul ----------------
// Worker task on core 0 computes the upper half of each matmul's rows
// while the inference task (core 1) computes the lower half.
static TaskHandle_t s_worker = nullptr;
static TaskHandle_t s_main   = nullptr;
static volatile const MMJob* s_job = nullptr;

static void worker_task(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);          // wait for a job
    const MMJob* j = (const MMJob*)s_job;
    int mid = j->d / 2;
    llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx,
                    j->n, mid, j->d);
    xTaskNotifyGive(s_main);                          // signal done
  }
}

static void parallel_matmul(const MMJob* j) {
  if (j->d < 64) {                                    // not worth the handoff
    llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, j->d);
    return;
  }
  s_job = j;
  xTaskNotifyGive(s_worker);                          // kick core 0
  int mid = j->d / 2;
  llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, mid);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);            // join
}

// ---------------- model ----------------
static LLM g_m = {};
static Sampler g_sampler = { 0.8f, 0.9f, 0 };
static bool g_ready = false;

static bool load_model() {
  const esp_partition_t* part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { ESP_LOGI(TAG,"ERROR: 'model' partition not found. Wrong partitions.csv?"); return false; }

  const void* base = nullptr;
  esp_partition_mmap_handle_t h;
  if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA,
                         &base, &h) != ESP_OK) {
    ESP_LOGI(TAG,"ERROR: flash mmap failed");
    return false;
  }
  ESP_LOGI(TAG,"Model partition mapped: %u bytes at %p\n", part->size, base);

  int rc = llm_init(&g_m, (const uint8_t*)base, part->size);
  if (rc) {
    ESP_LOGI(TAG,"ERROR: llm_init -> %d. Did you flash model_esp.bin to 0x%X?\n",
                  rc, part->address);
    return false;
  }
  if (llm_tok_init(&g_m)) { ESP_LOGI(TAG,"ERROR: tokenizer init"); return false; }

  ESP_LOGI(TAG,"Model: dim=%d layers=%d heads=%d vocab=%d seq=%d, INT%d gs=%d\n",
                g_m.h.dim, g_m.h.n_layers, g_m.h.n_heads, g_m.h.vocab_size,
                g_m.h.seq_len, g_m.h.bits, g_m.h.gs);
  ESP_LOGI(TAG,"Free heap: internal %u KB, PSRAM %u KB\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
  return true;
}

// ---------------- generation ----------------
static void generate(const char* prompt) {
  static int tokens[512];
  int n = llm_encode(&g_m, prompt, 1, 0, tokens, 512);
  if (n < 1) { ESP_LOGI(TAG,"(empty prompt)"); return; }
  if (n >= g_m.h.seq_len) { ESP_LOGI(TAG,"(prompt too long)"); return; }

  g_sampler.temperature = g_temp;
  g_sampler.topp = g_topp;

  ESP_LOGI(TAG,"\n> ");
  ESP_LOGI(TAG,"RAW HEX: %08X", *(unsigned int*)&prompt);
  uint32_t t0 = xTaskGetTickCount(); // 获取当前系统 Tick
  int generated = 0;
  int tok = tokens[0];
  #define MIN(a, b) ((a) < (b) ? (a) : (b))
    int total = MIN(n + g_maxlen, (int)g_m.h.seq_len);

  for (int pos = 0; pos < total; pos++) {
    llm_forward(&g_m, tok, pos);
    int next;
    if (pos < n - 1) next = tokens[pos + 1];           // prefill
    else { next = llm_sample(&g_m, &g_sampler); generated++; }
    if (next == 1 || next == 2) break;                 // BOS/EOS
    if (pos >= n - 1) {
    printf("%s", llm_decode(&g_m, tok, next));
    fflush(stdout); // 强行刷新控制台缓存，让文本一个词一个词实时蹦出来
}
    tok = next;
  }
  uint32_t dt = (xTaskGetTickCount() - t0) * portTICK_PERIOD_MS; // 转换为毫秒
  ESP_LOGI(TAG,"\n\n[%d tokens in %.1fs — %.2f tok/s]\n",
                generated, dt / 1000.0f, generated * 1000.0f / dt);
}

#include "esp_rom_serial_output.h" // ⚡ 引入这行！绕过操作系统的底层硬件接口

void serial_ui_task(void *pvParameters) {
    char line[256];
    int pos = 0;

    printf("\nType a prompt and press Enter.\nTLM> ");
    fflush(stdout);

    while (1) {
        if (!g_ready) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // 直接用 fgetc + 显式 delay，放弃 select
        // select 在 ESP-IDF UART stdin 上不可靠
        int ch = EOF;
        
        // 非阻塞读：先检查有没有数据
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(fileno(stdin), &read_fds);
        struct timeval tv = {0, 0};  // 零超时 = 纯轮询，不阻塞
        
    uint8_t ch_buf;
    int len = uart_read_bytes(UART_NUM_0, &ch_buf, 1, pdMS_TO_TICKS(20));
    if (len > 0) {
        int ch = ch_buf;
            putchar(ch);
            fflush(stdout);

            if (ch == '\n' || ch == '\r') {
                if (pos == 0) {
                    printf("\nTLM> ");
                    fflush(stdout);
                } else {
                    line[pos] = '\0';
                    pos = 0;
                    printf("\n");
                    
                    if (line[0] == '/') {
                     float fv; int iv;                        
                    if (sscanf(line, "/temp %f", &fv) == 1) { g_temp = fv; printf("temp=%.2f\n", fv); }                        
                    else if (sscanf(line, "/topp %f", &fv) == 1) { g_topp = fv; printf("topp=%.2f\n", fv); }                        
                    else if (sscanf(line, "/len %d", &iv) == 1) { g_maxlen = iv; printf("len=%d\n", iv); }                        
                    else if (strcmp(line, "/stats") == 0) {                            
                        printf("internal %u KB free, PSRAM %u KB free, temp=%.2f topp=%.2f len=%d\n",                                   
                            heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,                                   
                            heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024,                                   
                            g_temp, g_topp, g_maxlen);
                        }
                    else printf("commands: /temp /topp /len /stats\n");
                    } else {
                        generate(line);
                    }
                    printf("\nTLM> ");
                    fflush(stdout);
                }
            } else if (ch == 127 || ch == '\b') {
                if (pos > 0) { pos--; printf("\b \b"); fflush(stdout); }
            } else if (pos < 255) {
                line[pos++] = ch;
            }
        }

        // 关键：无论有没有读到字符，每次循环都显式让出 CPU
        // 20ms 足够响应键盘输入，也足够让 IDLE 任务喂狗
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
void app_main(void) {
    uart_init();
    // 1. 初始化板载 LED (对应原 pinMode(2, OUTPUT))
    gpio_reset_pin(GPIO_NUM_2);
    gpio_set_direction(GPIO_NUM_2, GPIO_MODE_OUTPUT);
    gpio_set_level(GPIO_NUM_2, 1);
    
    // 延时 1.5 秒
    vTaskDelay(pdMS_TO_TICKS(1500));

    // 2. 用标准的 printf 打印 UI 头部
    printf("\n==============================================\n");
    printf("  ESP32-S3 Storyteller — on-device LLM\n");
    printf("  INT4/8 weights in flash, dual-core matmul\n");
    printf("==============================================\n");

    // 3. 检查 PSRAM 状态
    multi_heap_info_t info;
    heap_caps_get_info(&info, MALLOC_CAP_SPIRAM);
    if (info.total_free_bytes == 0) {
        ESP_LOGE(TAG, "ERROR: PSRAM not found/enabled! Check sdkconfig.");
        return;
    }

    // 获取当前 main 任务的句柄
    s_main = xTaskGetCurrentTaskHandle();

    // 4. 创建你的双核矩阵乘法 Worker 任务 (绑定到 Core 0)
    xTaskCreatePinnedToCore(worker_task, "mm_worker", 4096, NULL,
                            configMAX_PRIORITIES - 2, &s_worker, 0);

    // 缝合你的底层函数指针（保持你原有的逻辑）
    // llm_alloc_big = alloc_big; ...

    // 5. 载入模型
    g_ready = load_model();
    
    if (g_ready) {
        // 使用 ESP32 硬件随机数发生器初始化 RNG
        // g_sampler.rng = esp_random() | 1ULL;
        
        printf("\nType a story opening (e.g. \"Once upon a time\") and press Enter.\n");
        printf("Commands: /temp X  /topp X  /len N  /stats\n\n");
    }

    // 6. 启动控制台 UI 任务 (建议绑定到 Core 1，与通信/计算隔离)
    xTaskCreatePinnedToCore(serial_ui_task, "serial_ui", 4096, NULL, 5, NULL, 1);
}