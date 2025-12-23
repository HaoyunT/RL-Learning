#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "rknn_api.h"

/* ===== 根据你的模型实际情况改这两个 ===== */
#define OBS_DIM 26
#define ACT_DIM 2
#define LOOP_US 10000   // 10ms -> 100Hz

/* 读取 rknn 文件 */
static unsigned char* load_model(const char* path, int* size)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror("open rknn");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);

    unsigned char* data = (unsigned char*)malloc(*size);
    fread(data, 1, *size, fp);
    fclose(fp);
    return data;
}

int main()
{
    const char* model_path = "./actor_agent0.rknn";

    printf("========== RL Actor on RK3588 (Deployment) ==========\n");

    /* ===== 1. 加载模型 ===== */
    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (!model_data) {
        return -1;
    }

    /* ===== 2. 初始化 RKNN ===== */
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);

    if (ret != RKNN_SUCC) {
        printf("rknn_init failed: %d\n", ret);
        return -1;
    }

    printf("RKNN init success\n");

    /* ===== 3. 准备输入输出 buffer ===== */
    float obs[OBS_DIM];
    float action[ACT_DIM];

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type  = RKNN_TENSOR_FLOAT32;
    input.size  = sizeof(obs);
    input.buf   = obs;

    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.want_float = 1;
    printf("Start inference loop (100Hz)...\n");

    /* ===== 4. 主循环：这就是"部署" ===== */
    float t = 0.f;  // 在循环外部声明时间变量
    
    while (1) {

        /* ---- mock 输入（现在先用假数据） ---- */
        for (int i = 0; i < OBS_DIM; i++) {
            obs[i] = 0.01f * i+0.1 * sinf(t);
        }

        /* ---- 输出观测数据 ---- */
        printf("obs = [");
        for (int i = 0; i < OBS_DIM; i++) {
            printf("%.4f", obs[i]);
            if (i < OBS_DIM-1) printf(", ");
        }
        printf("]\n");

        /* ---- 推理 ---- */
        rknn_inputs_set(ctx, 1, &input);
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, 1, &output, NULL);

        memcpy(action, output.buf, sizeof(float) * ACT_DIM);
        rknn_outputs_release(ctx, 1, &output);

        /* ---- 输出 action ---- */
        printf("action = [%.4f, %.4f]\n", action[0], action[1]);

        /* ---- 更新时间 ---- */
        t += 0.1f;  // 每次循环增加0.1，让输入数据随时间变化

        /* ---- 控制频率 ---- */
        usleep(LOOP_US);
    }

    rknn_destroy(ctx);
    return 0;
}
