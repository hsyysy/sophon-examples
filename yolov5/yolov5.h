#ifdef __ARM_NEON
#define STBI_NEON
#endif

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#endif
#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif

#include <sys/stat.h>
#include <math.h>
#include <string.h>

#ifndef YOLOV5_H
#define YOLOV5_H
#include "text2img.h"
#include "utils.h"

const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

void pre_process(const unsigned char* img, float* input_data, struct resize_info* r){
    int channels = 3;

    // using stb_image_resize to resize
    int target_w = r->net_w, target_h = r->net_h;
    if (r->keep_aspect){
        if (r->ratio_x < r->ratio_y){
            target_h = (int)(r->ori_h * r->ratio_x);
            r->start_y = (int)(r->net_h-target_h)/2;
            r->ratio_y = r->ratio_x;
        } else {
            target_w = (int)(r->ori_w * r->ratio_y);
            r->start_x = (int)(r->net_w-target_w)/2;
            r->ratio_x = r->ratio_y;
        }
    }
    unsigned char *resized_img = (unsigned char *)malloc(target_w * target_h * channels);
    stbir_resize_uint8_linear(img, r->ori_w, r->ori_h, 0, resized_img, target_w, target_h, 0, STBIR_RGB);
    //stbi_write_bmp("check.bmp", target_w, target_h, channels, (void*)resized_img);

    // fill the input_data from resized_img
    // input data is CHW, but resized_img is HWC
    float* input_temp0 = input_data + r->start_y * r->net_w + r->start_x;
    unsigned temp_w = target_w*channels;
    int net_area = r->net_w * r->net_h;
    for (int k=0;k<channels;k++){
        float* input_temp1 = input_temp0 + k*net_area;
        unsigned char* r_temp1 = resized_img + k;
        for (int i=0;i<target_h;i++){
            float* input_temp2 = input_temp1 + i*r->net_w;
            unsigned char* r_temp2 = r_temp1 + i*temp_w;
            for (int j=0;j<target_w;j++){
                input_temp2[j] = (float)r_temp2[j*channels]/255.0;
            }
        }
    }

    free(resized_img);
}

void post_process(float** output, const char* img_path, unsigned char* img,
        struct resize_info* r_info){
    float m_confThreshold = 0.5;

    int anchors[3][3][2] = {
        {{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}};
    int box_size[3] = {80,40,20};
    const int anchor_num = 3;
    int output_num = 3;
    int box_num = 25200; // 3*(80*80+40*40+20*20)
    int nout = 85;
    float* data = (float*)malloc(box_num*nout*sizeof(float));
    float* dst = data;

    for(int tidx = 0; tidx < output_num; ++tidx) {
        int feat_h = box_size[tidx];
        int feat_w = box_size[tidx];
        int area = feat_h * feat_w;
        int feature_size = area*nout;
        for (int anchor_idx = 0; anchor_idx < anchor_num; anchor_idx++) {
            float* ptr = output[tidx] + anchor_idx*feature_size;
            for (int i = 0; i < area; i++) {
                dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * r_info->net_w;
                dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * r_info->net_h;
                dst[2] = pow((sigmoid(ptr[2]) * 2), 2) * anchors[tidx][anchor_idx][0];
                dst[3] = pow((sigmoid(ptr[3]) * 2), 2) * anchors[tidx][anchor_idx][1];
                dst[4] = sigmoid(ptr[4]);
                float score = dst[4];
                if (score > m_confThreshold) {
                    for(int d=5; d<nout; d++){
                        dst[d] = sigmoid(ptr[d]);
                    }
                }
                dst += nout;
                ptr += nout;
            }
        }
    }

    int m_class_num = 80;
    struct YoloV5Box* yolobox = (struct YoloV5Box*)malloc( box_num * sizeof(struct YoloV5Box));
    int box_i = 0;
    for (int i = 0; i < box_num; i++) {
        float* ptr = data+i*nout;
        float score = ptr[4];
        if (score > m_confThreshold) {
            unsigned class_id = 0;
            float confidence = ptr[5];
#if defined(__ARM_NEON)
            argmax_neon(&ptr[5], m_class_num, &confidence, &class_id);
#elif defined(__SSE4_1__)
            argmax_sse(&ptr[5], m_class_num, &confidence, &class_id);
#else
            argmax(&ptr[5], m_class_num, &confidence, &class_id);
#endif
            float final_score = confidence * score;
            if (final_score > m_confThreshold) {
                struct YoloV5Box* box = &yolobox[box_i];
                float w = ptr[2];
                float h = ptr[3];
                box->x = ptr[0] - w / 2;
                box->y = ptr[1] - h / 2;
                box->w = w;
                box->h = h;
                box->class_id = class_id;
                box->score    = final_score;

                box_i ++;
            }
        }
    }
    free(data);

    // doing NMS
    float nmsConfidence = 0.6;
    bool* keep = (bool*)malloc(box_i*sizeof(bool));
    memset(keep, true, box_i*sizeof(bool));
    NMS(yolobox, keep, nmsConfidence, box_i);
    unsigned box_id = 0;
    size_t colors_num = sizeof(colors)/3/sizeof(int);
    // plot the rect on the img
    for (int i=0;i<box_i;i++){
        if (keep[i]){
            struct YoloV5Box* box = &yolobox[i];
            box->x = (box->x - r_info->start_x) / r_info->ratio_x;
            box->y = (box->y - r_info->start_y) / r_info->ratio_y;
            box->w = box->w / r_info->ratio_x;
            box->h = box->h / r_info->ratio_y;
            fix_box(box,r_info->ori_w,r_info->ori_h);
            int color_id = box->class_id % colors_num;
            draw_rect(img,box,r_info->ori_w,colors[color_id]);
            put_text(img, r_info->ori_w, r_info->ori_h, CLASS_NAMES[box->class_id], box->x, box->y, 0.5);
            printf("class[%02d]: scores = %f, label = %s\n", box_id++,box->score,CLASS_NAMES[box->class_id]);
        }
    }
    free(keep);

    // check whether results directory exists
    struct stat st = {0};
    if (stat("results", &st) == -1) {
        if (mkdir("results", 0700) == 0) {
            printf("Directory 'results' created successfully.\n");
        } else {
            perror("Error creating directory");
        }
    }

    // save result bmp
    char result_name[256];
    char filename_without_extension[256];
    get_filename_without_extension(img_path, filename_without_extension);
    if (result_name != NULL) {
        strcpy(result_name, "results/");
        strcat(result_name, filename_without_extension);
        strcat(result_name, ".bmp");
    }
    stbi_write_bmp(result_name, r_info->ori_w, r_info->ori_h, 3, (void*)img);
    printf("Save result bmp to : %s\n", result_name);

    // free result box struct
    free(yolobox);
}
#endif
