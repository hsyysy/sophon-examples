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
    unsigned class_id;
};

// sigmoid function
float sigmoid(float x){
    return 1.0 / (1 + expf(-x));
}

// argmax function
void argmax(const float* data, int num, float* max_value, unsigned* max_index){
    for(int i = 1; i < num; ++i) {
        float value = data[i];
        if (value > *max_value) {
            *max_value = value;
            *max_index = i;
        }
  }
}

#ifdef __SSE4_1__
void argmax_sse(const float* data, int num, float* max_value, unsigned *max_index)
{
    float aMaxVal[4];
    int32_t aMaxIndex[4];
    int i;

    const __m128i vIndexInc = _mm_set1_epi32(4);
    __m128i vMaxIndex = _mm_setr_epi32(0, 1, 2, 3);
    __m128i vIndex = vMaxIndex;
    __m128 vMaxVal = _mm_loadu_ps(data);

    for (i = 4; i < num; i += 4)
    {
        __m128 v = _mm_loadu_ps(&data[i]);
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
void argmax_neon(const float *data, int num, float* max_value, unsigned* max_index) {
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

float calculate_iou(struct YoloV5Box* box1, struct YoloV5Box* box2, float* area1, float* area2) {
    float x1 = fmaxf(box1->x, box2->x);
    float y1 = fmaxf(box1->y, box2->y);
    float x2 = fminf(box1->x+box1->width,  box2->x+box2->width);
    float y2 = fminf(box1->y+box1->height, box2->y+box2->height);

    float intersection = fmaxf(0.0f, x2 - x1 + 0.00001f) * fmaxf(0.0f, y2 - y1 + 0.00001f);

    return intersection / (*area1 + *area2 - intersection);
}

void NMS(struct YoloV5Box* dets, bool* keep, float nmsConfidence, int length){
    float* areas = (float*)malloc(length*sizeof(float));
    for (int i=0; i<length; i++) {
        areas[i] = dets[i].width * dets[i].height;
    }
    for (int i=0;i < length; i++){
        if (!keep[i]) continue;
        for (int j = i + 1; j < length; j++) {
            if (!keep[j]) continue;
            float iou = calculate_iou(dets + i, dets + j, areas + i, areas + j);
            if (iou > nmsConfidence) {
                if (dets[i].score > dets[j].score) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                }
            }
        }
    }
    free(areas);
}

// draw rect on img
void draw_rect(unsigned char* img, const struct YoloV5Box* box,
        const unsigned width, const unsigned height, const int* color){
    int x = (int)box->x;
    int y = (int)box->y;
    int w = (int)box->width;
    int h = (int)box->height;
    for (int j=0;j<w;j++){
        img[3*( y   *width+x+j)  ] = color[0];
        img[3*( y   *width+x+j)+1] = color[1];
        img[3*( y   *width+x+j)+2] = color[2];
        img[3*((y+h)*width+x+j)  ] = color[0];
        img[3*((y+h)*width+x+j)+1] = color[1];
        img[3*((y+h)*width+x+j)+2] = color[2];
    }
    for (int j=0;j<h;j++){
        img[3*((y+j)*width+x)    ] = color[0]; // R
        img[3*((y+j)*width+x)  +1] = color[1]; // G
        img[3*((y+j)*width+x)  +2] = color[2]; // B
        img[3*((y+j)*width+x+w)  ] = color[0]; // R
        img[3*((y+j)*width+x+w)+1] = color[1]; // G
        img[3*((y+j)*width+x+w)+2] = color[2]; // B
    }
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
