#include <float.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// rect color list
const int colors[25][3] = {
    {255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, \
    {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, \
    {0, 170, 255}, {0, 85, 255}, {0, 0, 255}, {85, 0, 255}, {170, 0, 255}, \
    {255, 0, 255}, {255, 0, 170}, {255, 0, 85}, {255, 0, 0},{255, 0, 255}, \
    {255, 85, 255}, {255, 170, 255}, {255, 255, 255}, {170, 255, 255}, {85, 255, 255}
};

// result struct
struct YoloV5Box {
    float x, y, width, height;
    float score;
    int class_id;
};

// sigmoid function
float sigmoid(float x){
    return 1.0 / (1 + expf(-x));
}

// argmax function
void argmax(const float* data, int num, float* max_value, int* max_index){
    for(int i = 1; i < num; ++i) {
        float value = data[i];
        if (value > *max_value) {
            *max_value = value;
            *max_index = i;
        }
  }
}

#ifdef __SSE4_1__
void argmax_sse(const float* m, int num, float* max_value, int *max_index)
{
    float aMaxVal[4];
    int32_t aMaxIndex[4];
    int i;

    const __m128i vIndexInc = _mm_set1_epi32(4);
    __m128i vMaxIndex = _mm_setr_epi32(0, 1, 2, 3);
    __m128i vIndex = vMaxIndex;
    __m128 vMaxVal = _mm_loadu_ps(m);

    for (i = 4; i < num; i += 4)
    {
        __m128 v = _mm_loadu_ps(&m[i]);
        __m128 vcmp = _mm_cmpgt_ps(v, vMaxVal);
        vIndex = _mm_add_epi32(vIndex, vIndexInc);
        vMaxVal = _mm_max_ps(vMaxVal, v);
        vMaxIndex = _mm_blendv_epi8(vMaxIndex, vIndex, _mm_castps_si128(vcmp));
    }
    _mm_storeu_ps(aMaxVal, vMaxVal);
    _mm_storeu_si128((__m128i *)aMaxIndex, vMaxIndex);
    *max_value = aMaxVal[0];
    *max_index = aMaxIndex[0];
    for (i = 1; i < 4; ++i)
    {
        if (aMaxVal[i] > *max_value)
        {
            *max_value = aMaxVal[i];
            *max_index = aMaxIndex[i];
        }
    }
}
#endif

#ifdef __ARM_NEON
void argmax_neon(const float *data, int num, float* max_value, int* max_index) {
    // 初始化最大值向量和索引
    float32x4_t max_val_vec = vdupq_n_f32(-FLT_MAX);
    uint32x4_t max_idx_vec = vdupq_n_u32(0);

    // 当前处理的基准索引
    uint32x4_t base_idx_vec = {0, 1, 2, 3};
    // 索引增量
    uint32x4_t idx_increment = vdupq_n_u32(4);

    // 用于跟踪全局最大值和索引
    float max_val = -FLT_MAX;
    int max_idx = -1;

    int i;
    for (i = 0; i <= num - 4; i += 4) {
        // 加载数据
        float32x4_t data_vec = vld1q_f32(data + i);

        // 比较向量中的元素，更新最大值和索引
        uint32x4_t mask = vcgtq_f32(data_vec, max_val_vec);
        max_val_vec = vbslq_f32(mask, data_vec, max_val_vec);
        max_idx_vec = vbslq_u32(mask, base_idx_vec, max_idx_vec);

        // 更新基准索引
        base_idx_vec = vaddq_u32(base_idx_vec, idx_increment);
    }

    // 处理剩余的元素
    for (; i < num; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }

    // 将向量中的最大值和索引提取到标量中
    float max_vals[4];
    uint32_t max_idxs[4];
    vst1q_f32(max_vals, max_val_vec);
    vst1q_u32(max_idxs, max_idx_vec);

    // 在向量结果中找到最终的最大值和索引
    for (int j = 0; j < 4; j++) {
        if (max_vals[j] > max_val) {
            max_val = max_vals[j];
            max_idx = max_idxs[j];
        }
    }

    // 返回结果
    *max_value = max_val;
    *max_index = max_idx;
}
#endif

// compare two struct scores
int compare(const void* a, const void* b) {
    struct YoloV5Box* structA = (struct YoloV5Box*)a;
    struct YoloV5Box* structB = (struct YoloV5Box*)b;

    if (structA->score < structB->score) {
        return 1;  // 返回正值，表示 structA 应该排在 structB 后面
    } else if (structA->score > structB->score) {
        return -1; // 返回负值，表示 structA 应该排在 structB 前面
    } else {
        return 0;  // 返回 0，表示两者相等
    }
}

// remove an element from data
void removeElement(void* arr, int* n, int i, size_t elem_size) {
    // make sure i is valid
    if (i < 0 || i >= *n) {
        printf("Invalid index\n");
        return;
    }

    // move ahead for elements after i
    for (int j = i; j < *n - 1; j++) {
        // using memmove to avoid memory overlap
        memmove((char*)arr + j * elem_size, (char*)arr + (j + 1) * elem_size, elem_size);
    }

    // renew n
    (*n)--;

    // re-allocate memory
    arr = realloc(arr, (*n) * elem_size);
    if (arr == NULL && *n > 0) {
        printf("Memory reallocation failed\n");
        exit(1);
    }
}

// NMS function
int NMS(struct YoloV5Box* dets, float nmsConfidence, int length) {
    int num = length;
    int index = length - 1;

    qsort(dets, length, sizeof(struct YoloV5Box), compare);

    float* areas = (float*)malloc(length*sizeof(float));
    for (int i=0; i<length; i++) {
        areas[i] = dets[i].width * dets[i].height;
    }

    int n1 = length;
    int n2 = length;
    while (index  > 0) {
        int i = 0;
        while (i < index) {
            float left    = fmax(dets[index].x,  dets[i].x);
            float top     = fmax(dets[index].y,  dets[i].y);
            float right   = fmin(dets[index].x + dets[index].width,  dets[i].x + dets[i].width);
            float bottom  = fmin(dets[index].y + dets[index].height, dets[i].y + dets[i].height);
            float overlap = fmax(0.0f, right - left + 0.00001f) * fmax(0.0f, bottom - top + 0.00001f);
            if (overlap / (areas[index] + areas[i] - overlap) > nmsConfidence) {
                removeElement(areas, &n1, i,sizeof(areas[0]));
                removeElement(dets, &n2, i,sizeof(dets[0]));
                num --;
                index --;
            } else {
                i++;
            }
        }
        index--;
    }
    free(areas);
    return num;
}

// get filename without extension
void get_filename_without_extension(const char *path, char *output) {
    // remove path to get filename
    const char *filename = strrchr(path, '/');  // UNIX-like system use '/'
    if (filename == NULL) {
        filename = strrchr(path, '\\');  // for Windows using '\\'
    }

    // if there is no path sep, then the total sting is filename
    if (filename == NULL) {
        filename = path;
    } else {
        filename++;  // skip path sep
    }

    // remove the extension
    const char *dot = strrchr(filename, '.');
    if (dot != NULL) {
        size_t len = dot - filename;
        strncpy(output, filename, len);
        output[len] = '\0';
    } else {
        // if there is no extension, then copy the total filename
        strcpy(output, filename);
    }
}
