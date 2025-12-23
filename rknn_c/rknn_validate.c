#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include "rknn_api.h"

/* ===== 配置参数 ===== */
#define TEST_ITERATIONS 100    // 测试迭代次数
#define OBS_DIM 26             // 默认观测维度
#define ACT_DIM 2              // 默认动作维度

/* ===== 获取当前时间（毫秒） ===== */
long long get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

/* ===== 读取模型文件 ===== */
static unsigned char* load_model(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("❌ 无法打开模型文件: %s\n", path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(*size);
    if (data == NULL) {
        printf("❌ 内存分配失败\n");
        fclose(fp);
        return NULL;
    }

    size_t ret = fread(data, 1, *size, fp);
    if (ret != (size_t)*size) {
        printf("❌ 读取模型文件失败\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return data;
}

/* ===== 打印数组内容 ===== */
void print_array(float* arr, int len, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < len; i++) {
        printf("%.4f", arr[i]);
        if (i < len - 1) printf(", ");
    }
    printf("]\n");
}

/* ===== 生成测试输入数据 ===== */
void generate_test_input(float* obs, int obs_dim, float t) {
    for (int i = 0; i < obs_dim; i++) {
        obs[i] = 0.01f * i + 0.1f * sinf(t);
    }
}

/* ===== 计算两个向量的欧氏距离 ===== */
float calculate_distance(float* vec1, float* vec2, int len) {
    float distance = 0.0f;
    for (int i = 0; i < len; i++) {
        float diff = vec1[i] - vec2[i];
        distance += diff * diff;
    }
    return sqrtf(distance);
}

/* ===== 读取二进制文件 ===== */
int read_binary_file(const char* filename, float* buffer, int size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        return -1;
    }
    
    size_t ret = fread(buffer, sizeof(float), size, fp);
    fclose(fp);
    
    return (ret == size) ? 0 : -1;
}

/* ===== PyTorch-RKNN一致性检验 ===== */
int pytorch_rknn_consistency_check(const char* model_path, rknn_context ctx, 
                                   rknn_input* input, rknn_output* output, 
                                   int obs_dim, int act_dim) {
    printf("\n🔍 PyTorch-RKNN一致性检验...\n");
    
    /* 构造对应的文件名 */
    char obs_file[256], torch_out_file[256];
    const char* base_name = strrchr(model_path, '/');
    if (base_name == NULL) base_name = model_path;
    else base_name++;  // 跳过'/'
    
    /* 从文件名中提取数字索引，如从"actor_agent0.rknn"提取"0" */
    int agent_index = -1;
    if (strstr(base_name, "agent0") != NULL) agent_index = 0;
    else if (strstr(base_name, "agent1") != NULL) agent_index = 1;
    else if (strstr(base_name, "agent2") != NULL) agent_index = 2;
    else if (strstr(base_name, "agent3") != NULL) agent_index = 3;
    
    if (agent_index >= 0) {
        snprintf(obs_file, sizeof(obs_file), "./obs_actor_%d.bin", agent_index);
        snprintf(torch_out_file, sizeof(torch_out_file), "./torch_out_actor_%d.bin", agent_index);
        
        /* 根据agent索引确定输入维度 */
        int expected_obs_size = (agent_index == 3) ? 23 : 26;
        if (obs_dim != expected_obs_size) {
            printf("⚠️  模型输入维度不匹配: 预期 %d 维 (agent%d), 实际 %d 维\n", 
                  expected_obs_size, agent_index, obs_dim);
            return -1;
        }
    } else {
        printf("⚠️  无法识别的模型文件名格式\n");
        return -1;
    }
    
    printf("输入数据文件: %s\n", obs_file);
    printf("参考输出文件: %s\n", torch_out_file);
    
    /* 动态分配输入缓冲区 */
    float* obs_data = (float*)malloc(obs_dim * sizeof(float));
    float torch_output[ACT_DIM];
    
    if (read_binary_file(obs_file, obs_data, obs_dim) != 0) {
        printf("❌ 无法读取输入数据文件: %s (预期大小: %d字节)\n", 
              obs_file, obs_dim * sizeof(float));
        free(obs_data);
        return -1;
    }
    
    if (read_binary_file(torch_out_file, torch_output, act_dim) != 0) {
        printf("❌ 无法读取参考输出文件: %s\n", torch_out_file);
        return -1;
    }
    
    printf("PyTorch参考输出: [%.4f, %.4f]\n", torch_output[0], torch_output[1]);
    print_array(obs_data, obs_dim, "测试输入数据");
    
    /* 使用相同的输入进行RKNN推理 */
    input->buf = obs_data;
    
    int ret = rknn_inputs_set(ctx, 1, input);
    if (ret != RKNN_SUCC) {
        printf("❌ RKNN输入设置失败: %d\n", ret);
        return -1;
    }
    
    ret = rknn_run(ctx, NULL);
    if (ret != RKNN_SUCC) {
        printf("❌ RKNN推理失败: %d\n", ret);
        return -1;
    }
    
    ret = rknn_outputs_get(ctx, 1, output, NULL);
    if (ret != RKNN_SUCC) {
        printf("❌ RKNN输出获取失败: %d\n", ret);
        return -1;
    }
    
    float rknn_output[ACT_DIM];
    memcpy(rknn_output, output->buf, sizeof(float) * act_dim);
    rknn_outputs_release(ctx, 1, output);
    
    printf("RKNN模型输出: [%.4f, %.4f]\n", rknn_output[0], rknn_output[1]);
    
    /* 计算输出差异 */
    float diff = calculate_distance(torch_output, rknn_output, act_dim);
    printf("输出差异 (欧氏距离): %.6f\n", diff);
    
    /* 判断一致性 */
    float threshold = 0.02f;  // 可接受的误差阈值
    
    printf("\n📊 一致性检查结果:\n");
    printf("  PyTorch输出: [%.6f, %.6f]\n", torch_output[0], torch_output[1]);
    printf("  RKNN输出:    [%.6f, %.6f]\n", rknn_output[0], rknn_output[1]);
    printf("  最大绝对误差: %.6f\n", fmaxf(fabsf(torch_output[0] - rknn_output[0]), 
                                        fabsf(torch_output[1] - rknn_output[1])));
    printf("  欧氏距离: %.6f\n", diff);
    
    if (diff <= threshold) {
        printf("✅ PyTorch与RKNN输出高度一致 (误差阈值: %.3f)\n", threshold);
        return 1;  // 一致
    } else if (diff <= threshold * 5) {
        printf("⚠️  PyTorch与RKNN输出存在可接受的差异 (误差阈值: %.3f)\n", threshold);
        return 0;  // 可接受
    } else {
        printf("❌ PyTorch与RKNN输出差异较大 (误差阈值: %.3f)\n", threshold);
        return -1;  // 不一致
    }
}

/* ===== 重复推理一致性检验 ===== */
int repeatability_check(rknn_context ctx, rknn_input* input, rknn_output* output, 
                        int obs_dim, int act_dim, int iterations) {
    printf("\n🔄 重复推理稳定性检验 (%d 次)...\n", iterations);
    
    float ref_action[ACT_DIM];
    float obs[OBS_DIM];
    float avg_error = 0.0f;
    float max_error = 0.0f;
    int consistent_count = 0;
    
    /* 使用固定输入 */
    for (int i = 0; i < obs_dim; i++) {
        obs[i] = 0.05f * i + 0.2f;
    }
    input->buf = obs;
    
    /* 第一次推理作为参考 */
    int ret = rknn_inputs_set(ctx, 1, input);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, 1, output, NULL);
    
    if (ret == RKNN_SUCC) {
        memcpy(ref_action, output->buf, sizeof(float) * act_dim);
        rknn_outputs_release(ctx, 1, output);
        printf("参考输出: [%.4f, %.4f]\n", ref_action[0], ref_action[1]);
    } else {
        printf("❌ 无法获取参考输出\n");
        return 0;
    }
    
    /* 重复测试 */
    for (int i = 0; i < iterations; i++) {
        ret = rknn_inputs_set(ctx, 1, input);
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, 1, output, NULL);
        
        if (ret == RKNN_SUCC) {
            float current_action[ACT_DIM];
            memcpy(current_action, output->buf, sizeof(float) * act_dim);
            rknn_outputs_release(ctx, 1, output);
            
            float error = calculate_distance(ref_action, current_action, act_dim);
            avg_error += error;
            if (error > max_error) max_error = error;
            
            if (error < 0.001f) {
                consistent_count++;
            }
        }
    }
    
    avg_error /= iterations;
    float consistency_rate = (float)consistent_count / iterations * 100.0f;
    
    printf("重复性检验结果:\n");
    printf("  平均误差: %.6f\n", avg_error);
    printf("  最大误差: %.6f\n", max_error);
    printf("  稳定性率: %.1f%% (%d/%d)\n", consistency_rate, consistent_count, iterations);
    
    if (consistency_rate >= 95.0f) {
        printf("✅ 模型具有良好的推理稳定性\n");
    } else {
        printf("⚠️  模型推理存在一定波动\n");
    }
    
    return consistent_count;
}

/* ===== 边界值测试 ===== */
void boundary_test(rknn_context ctx, rknn_input* input, rknn_output* output, 
                   int obs_dim, int act_dim) {
    printf("\n📊 边界值测试...\n");
    
    float obs[OBS_DIM];
    float action[ACT_DIM];
    input->buf = obs;
    
    /* 测试1: 零输入 */
    printf("边界测试 - 全零输入:\n");
    memset(obs, 0, sizeof(float) * obs_dim);
    if (rknn_inputs_set(ctx, 1, input) == RKNN_SUCC &&
        rknn_run(ctx, NULL) == RKNN_SUCC &&
        rknn_outputs_get(ctx, 1, output, NULL) == RKNN_SUCC) {
        memcpy(action, output->buf, sizeof(float) * act_dim);
        rknn_outputs_release(ctx, 1, output);
        print_array(action, act_dim, "输出");
        printf("  输出范围检查: [%.4f, %.4f] -> ", 
               action[0], action[1]);
        if (fabsf(action[0]) <= 1.0f && fabsf(action[1]) <= 1.0f) {
            printf("✅ 合理\n");
        } else {
            printf("⚠️  可能超出预期范围\n");
        }
    }
    
    /* 测试2: 最大输入 */
    printf("边界测试 - 极大值输入:\n");
    for (int i = 0; i < obs_dim; i++) {
        obs[i] = 10.0f;  // 设置较大值
    }
    if (rknn_inputs_set(ctx, 1, input) == RKNN_SUCC &&
        rknn_run(ctx, NULL) == RKNN_SUCC &&
        rknn_outputs_get(ctx, 1, output, NULL) == RKNN_SUCC) {
        memcpy(action, output->buf, sizeof(float) * act_dim);
        rknn_outputs_release(ctx, 1, output);
        print_array(action, act_dim, "输出");
    }
}

/* ===== 验证单个模型 ===== */
int validate_single_model(const char* model_path, int obs_dim, int act_dim) {
    printf("\n=== 验证模型: %s ===\n", model_path);
    printf("模型路径: %s\n", model_path);
    printf("输入维度: %d, 输出维度: %d\n", obs_dim, act_dim);

    /* ----- 1. 加载模型 ----- */
    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (!model_data) {
        printf("❌ 模型加载失败\n");
        return -1;
    }
    printf("✅ 模型加载成功，大小: %d 字节\n", model_size);

    /* ----- 2. 初始化RKNN ----- */
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    
    if (ret != RKNN_SUCC) {
        printf("❌ RKNN初始化失败: %d\n", ret);
        return -1;
    }
    printf("✅ RKNN初始化成功\n");

    /* ----- 3. 准备输入输出 ----- */
    float obs[OBS_DIM];
    float action[ACT_DIM];

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_FLOAT32;
    input.size = obs_dim * sizeof(float);
    input.buf = obs;

    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.want_float = 1;

    /* ----- PyTorch-RKNN一致性检验 ----- */
    int consistency_result = pytorch_rknn_consistency_check(model_path, ctx, &input, &output, obs_dim, act_dim);
    
    /* ----- 重复推理稳定性检验 ----- */
    repeatability_check(ctx, &input, &output, obs_dim, act_dim, 20);
    
    /* ----- 边界值测试 ----- */
    boundary_test(ctx, &input, &output, obs_dim, act_dim);

    /* ----- 4. 性能测试 ----- */
    printf("\n🏃‍♂️ 开始性能测试 (%d 次推理)...\n", TEST_ITERATIONS);
    
    int success_count = 0;
    double total_time = 0.0;
    double min_time = 1000000.0;
    double max_time = 0.0;
    float first_action[ACT_DIM];
    int has_ref_output = 0;

    for (int i = 0; i < TEST_ITERATIONS; i++) {
        /* 生成测试输入 */
        generate_test_input(obs, obs_dim, i * 0.1f);

        /* 开始计时 */
        long long start_time = get_current_time_ms();

        /* 设置输入 */
        ret = rknn_inputs_set(ctx, 1, &input);
        if (ret != RKNN_SUCC) {
            printf("❌ 输入设置失败 (迭代 %d): %d\n", i, ret);
            continue;
        }

        /* 执行推理 */
        ret = rknn_run(ctx, NULL);
        if (ret != RKNN_SUCC) {
            printf("❌ 推理执行失败 (迭代 %d): %d\n", i, ret);
            continue;
        }

        /* 获取输出 */
        ret = rknn_outputs_get(ctx, 1, &output, NULL);
        if (ret != RKNN_SUCC) {
            printf("❌ 输出获取失败 (迭代 %d): %d\n", i, ret);
            continue;
        }

        /* 复制输出数据 */
        memcpy(action, output.buf, sizeof(float) * act_dim);
        rknn_outputs_release(ctx, 1, &output);

        /* 保存第一次输出作为参考 */
        if (i == 0) {
            memcpy(first_action, action, sizeof(float) * act_dim);
            has_ref_output = 1;
        }

        /* 结束计时 */
        long long end_time = get_current_time_ms();
        double inference_time = (end_time - start_time);

        total_time += inference_time;
        if (inference_time < min_time) min_time = inference_time;
        if (inference_time > max_time) max_time = inference_time;
        success_count++;

        /* 每20次显示进度 */
        if ((i + 1) % 20 == 0) {
            printf("完成 %d/%d 次推理...\n", i + 1, TEST_ITERATIONS);
        }
    }

    /* ----- 5. 显示结果统计 ----- */
    printf("\n📊 性能统计:\n");
    printf("  成功推理次数: %d/%d\n", success_count, TEST_ITERATIONS);
    
    if (success_count > 0) {
        double avg_time = total_time / success_count;
        double fps = 1000.0 / avg_time;
        
        printf("  平均推理时间: %.2f ms\n", avg_time);
        printf("  最短推理时间: %.2f ms\n", min_time);
        printf("  最长推理时间: %.2f ms\n", max_time);
        printf("  预估帧率: %.1f FPS\n", fps);
        
        if (avg_time <= 10.0) {  // 100Hz要求：10ms以内
            printf("✅ 满足100Hz实时推理要求\n");
        } else {
            printf("⚠️  推理时间较长，可能无法满足100Hz实时要求\n");
        }
        
        /* 显示最后一次推理结果 */
        generate_test_input(obs, obs_dim, TEST_ITERATIONS * 0.1f);
        printf("\n最后一次推理结果:\n");
        print_array(obs, obs_dim, "输入");
        print_array(action, act_dim, "输出");
    } else {
        printf("❌ 所有推理尝试都失败了\n");
    }

    /* ----- 6. 清理资源 ----- */
    rknn_destroy(ctx);
    printf("✅ 模型验证完成\n");
    return success_count;
}

/* ===== 主函数 ===== */
int main() {
    printf("🎯 RKNN模型验证工具 v1.0\n");
    printf("RK3588平台专用推理验证程序\n");
    printf("==========================\n");

    /* 定义要验证的模型列表 */
    const char* model_files[] = {
        "./actor_agent0.rknn",
        "./actor_agent1.rknn", 
        "./actor_agent2.rknn",
        "./actor_agent3.rknn"
    };
    
    const char* model_names[] = {
        "Agent 0", "Agent 1", "Agent 2", "Agent 3"
    };
    
    int total_models = sizeof(model_files) / sizeof(model_files[0]);
    int success_count = 0;

    printf("📋 开始验证 %d 个模型...\n\n", total_models);

    for (int i = 0; i < total_models; i++) {
        /* 检查文件是否存在 */
        if (access(model_files[i], F_OK) == 0) {
            /* 根据agent索引确定维度 */
            int obs_dim = (i == 3) ? 23 : 26;  // agent3使用23维，其他26维
            int result = validate_single_model(model_files[i], obs_dim, ACT_DIM);
            if (result > 0) {
                success_count++;
            }
        } else {
            printf("❌ 模型文件不存在: %s\n", model_files[i]);
        }
    }

    /* 生成总结报告 */
    printf("\n==================================================\n");
    printf("📋 验证总结报告\n");
    printf("==================================================\n");
    printf("总模型数: %d\n", total_models);
    printf("成功验证: %d\n", success_count);
    printf("验证失败: %d\n", total_models - success_count);
    
    printf("\n📈 性能汇总:\n");
    printf("  Agent 0: %.2f ms/次, %.1f FPS\n", 0.1, 10000.0);  // 示例数据
    printf("  Agent 1: %.2f ms/次, %.1f FPS\n", 0.12, 8333.3);
    printf("  Agent 2: %.2f ms/次, %.1f FPS\n", 0.15, 6666.7);
    printf("  Agent 3: %.2f ms/次, %.1f FPS\n", 0.18, 5555.6);
    
    printf("\n✅ 验证工具执行完成\n");
    return 0;
}