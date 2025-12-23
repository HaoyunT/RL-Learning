#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <signal.h>
#include <syslog.h>
#include <sys/stat.h>
#include "rknn_api.h"

/* ===== 配置参数 ===== */
#define OBS_DIM 26             // 观测维度
#define ACT_DIM 2              // 动作维度
#define LOOP_US 10000          // 10ms间隔 -> 100Hz
#define DAEMON_NAME "rk3588_npu_daemon"

/* ===== 全局变量 ===== */
static int running = 1;
rknn_context ctx;

/* ===== 信号处理函数 ===== */
void signal_handler(int sig) {
    switch(sig) {
        case SIGTERM:
        case SIGINT:
            syslog(LOG_INFO, "收到终止信号，正在关闭服务...");
            running = 0;
            break;
        default:
            break;
    }
}

/* ===== 设置信号处理 ===== */
void setup_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    // 忽略这些信号，防止守护进程被意外终止
    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
}

/* ===== 读取模型文件 ===== */
static unsigned char* load_model(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        syslog(LOG_ERR, "无法打开模型文件: %s", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(*size);
    if (data == NULL) {
        syslog(LOG_ERR, "内存分配失败");
        fclose(fp);
        return NULL;
    }

    size_t ret = fread(data, 1, *size, fp);
    if (ret != (size_t)*size) {
        syslog(LOG_ERR, "读取模型文件失败");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return data;
}

/* ===== 模拟观测数据生成 ===== */
void generate_observation(float* obs, int obs_dim, float t) {
    for (int i = 0; i < obs_dim; i++) {
        // 更平滑的观测数据生成
        float base_val = 0.01f * i;
        float variation = 0.1f * sinf(t + i * 0.1f);
        obs[i] = base_val + variation;
    }
}

/* ===== 初始化RKNN模型 ===== */
int initialize_npu_system() {
    const char* model_path = "/home/khadas/rknn_c/actor_agent0.rknn";  // 默认使用agent0
    
    syslog(LOG_INFO, "正在加载NPU模型: %s", model_path);
    
    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (!model_data) {
        syslog(LOG_ERR, "模型加载失败");
        return -1;
    }

    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    
    if (ret != RKNN_SUCC) {
        syslog(LOG_ERR, "RKNN初始化失败: %d", ret);
        return -1;
    }
    
    syslog(LOG_INFO, "NPU系统初始化成功，模型大小: %d 字节", model_size);
    return 0;
}

/* ===== 执行推理循环 ===== */
void inference_loop() {
    float obs[OBS_DIM];
    float action[ACT_DIM];
    int inference_count = 0;
    
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_FLOAT32;
    input.size = sizeof(obs);
    input.buf = obs;

    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.want_float = 1;

    float time_var = 0.0f;
    
    syslog(LOG_INFO, "开始推理循环 (100Hz)...");
    
    while (running) {
        /* 生成观测数据 */
        generate_observation(obs, OBS_DIM, time_var);
        
        /* 设置输入 */
        int ret = rknn_inputs_set(ctx, 1, &input);
        if (ret != RKNN_SUCC) {
            syslog(LOG_ERR, "输入设置失败: %d", ret);
            usleep(LOOP_US);
            continue;
        }

        /* 执行推理 */
        ret = rknn_run(ctx, NULL);
        if (ret != RKNN_SUCC) {
            syslog(LOG_ERR, "推理执行失败: %d", ret);
            usleep(LOOP_US);
            continue;
        }

        /* 获取输出 */
        ret = rknn_outputs_get(ctx, 1, &output, NULL);
        if (ret != RKNN_SUCC) {
            syslog(LOG_ERR, "输出获取失败: %d", ret);
            usleep(LOOP_US);
            continue;
        }

        /* 处理输出 */
        memcpy(action, output.buf, sizeof(float) * ACT_DIM);
        rknn_outputs_release(ctx, 1, &output);
        
        inference_count++;
        
        /* 实时输出观测和动作数据 */
        printf("OBS: [");
        for (int i = 0; i < OBS_DIM; i++) {
            printf("%.4f", obs[i]);
            if (i < OBS_DIM-1) printf(", ");
        }
        printf("]\n");
        
        printf("ACTION: [%.4f, %.4f]\n", action[0], action[1]);
        
        /* 同时记录到系统日志 */
        if (inference_count % 100 == 0) {
            syslog(LOG_INFO, "推理统计: %d次推理, 动作: [%.4f, %.4f]", 
                   inference_count, action[0], action[1]);
        }
        
        /* 更新时间变量 */
        time_var += 0.1f;
        
        /* 控制推理频率 */
        usleep(LOOP_US);
    }
}

/* ===== 清理资源 ===== */
void cleanup() {
    syslog(LOG_INFO, "清理NPU资源...");
    if (ctx) {
        rknn_destroy(ctx);
    }
    syslog(LOG_INFO, "服务正常退出");
}

/* ===== 守护进程主函数 ===== */
int main(int argc, char *argv[]) {
    /* 设置为守护进程 */
    pid_t pid, sid;
    
    pid = fork();
    if (pid < 0) {
        exit(EXIT_FAILURE);
    }
    
    if (pid > 0) {
        exit(EXIT_SUCCESS);  // 父进程退出
    }
    
    /* 创建新的会话 */
    sid = setsid();
    if (sid < 0) {
        exit(EXIT_FAILURE);
    }
    
    /* 改变工作目录 */
    if ((chdir("/")) < 0) {
        exit(EXIT_FAILURE);
    }
    
    /* 关闭标准文件描述符 */
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
    
    /* 设置umask */
    umask(0);
    
    /* 设置信号处理 */
    setup_signal_handlers();
    
    /* 打开系统日志 */
    openlog(DAEMON_NAME, LOG_PID, LOG_DAEMON);
    syslog(LOG_INFO, "RK3588 NPU推理守护进程启动");
    
    /* 初始化NPU系统 */
    if (initialize_npu_system() != 0) {
        syslog(LOG_ERR, "NPU系统初始化失败");
        cleanup();
        closelog();
        exit(EXIT_FAILURE);
    }
    
    /* 主推理循环 */
    inference_loop();
    
    /* 清理和退出 */
    cleanup();
    closelog();
    
    return EXIT_SUCCESS;
}