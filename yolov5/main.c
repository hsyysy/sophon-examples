#ifdef __ARM_NEON
#define STBI_NEON
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <bmruntime_interface.h>
#include <sys/stat.h>
#include "utils.h"

#define KEEP_ASPECT

int main(int argc, char** argv){
    // request bm_handle
    bm_handle_t bm_handle;
    bm_status_t status;
#if defined(__arm__) || defined(__aarch64__)
    status = bm_dev_request(&bm_handle, 0);
    assert(BM_SUCCESS == status);
#else
    unsigned dev_id = 0;
    int total_dev;
    bm_dev_getcount(&total_dev);
    //printf("Total devices num = %d\n",total_dev);
    for (;dev_id < total_dev;dev_id++){
        status = bm_dev_request(&bm_handle, dev_id);
        assert(BM_SUCCESS == status);

        unsigned p_chipid;
        bm_get_chipid(bm_handle, &p_chipid);
        if (p_chipid == 0x1684){
            //printf("chip = BM1684\n");
            bm_dev_free(bm_handle);
            if (dev_id == total_dev-1){
                printf("There is no BM1684X chip!\n");
                exit(1);
            }
            continue;
        } else if (p_chipid == 0x1686){
            //printf("chip = BM1684X\n");
            printf("Select dev_id = %d with ",dev_id);
            break;
        }
    }
#endif

    // determine whether is soc
    struct bm_misc_info misc_info;
    status = bm_get_misc_info(bm_handle, &misc_info);
    assert(BM_SUCCESS == status);
    bool is_soc = misc_info.pcie_soc_mode;
    if (is_soc){
        printf("SOC Mode\n");
    } else {
        printf("PCIE Mode\n");
    }

    // create bmruntime
    void *p_bmrt = bmrt_create(bm_handle);
    assert(NULL != p_bmrt);

    // load bmodel by file
    bool ret = bmrt_load_bmodel(p_bmrt, "yolov5s_v6.1_3output_int8_1b.bmodel");
    assert(true == ret);

    // get net_info
    const char** net_names = NULL;
    bmrt_get_network_names(p_bmrt, &net_names);
    const bm_net_info_t* net_info = bmrt_get_network_info(p_bmrt, net_names[0]);
    assert(NULL != net_info);

    // get img path
    const char* img_path;
    if (argc > 1){
        img_path = argv[1];
    } else {
        img_path = "../datasets/dog.jpg";
    }

    // read image
    int width, height, channels;
    unsigned char *img = stbi_load(img_path, &width, &height, &channels, 0);
    if (img == NULL) {
            printf("Error in loading the image\n");
            exit(1);
    }
    printf("img: %s, width = %d, height = %d, channels = %d\n", img_path, width, height, channels);

    // initialize resized_img
    int net_h = net_info->stages[0].input_shapes->dims[2];
    int net_w = net_info->stages[0].input_shapes->dims[3];
    int net_area = net_w*net_h;
    unsigned char *resized_img = (unsigned char *)malloc(net_area * channels);
    if (resized_img == NULL) {
        printf("Unable to allocate memory for the resized image.\n");
        stbi_image_free(img);
        exit(1);
    }

    // using stb_image_resize to resize
    int target_w = net_w, target_h = net_h;
    float ratiox = (float)net_w/width;
    float ratioy = (float)net_h/height;
    int start_x = 0, start_y = 0;
#ifdef KEEP_ASPECT
    if (ratiox < ratioy){
        target_h = (int)(height * ratiox);
        start_y = (int)(net_h-target_h)/2;
        ratioy = ratiox;
    } else {
        target_w = (int)(width * ratioy);
        start_x = (int)(net_w-target_w)/2;
        ratiox = ratioy;
    }
#endif
    if (!stbir_resize_uint8_linear(img, width, height, 0, resized_img, target_w, target_h, 0, STBIR_RGB)) {
            printf("Failed to resize image\n");
            stbi_image_free(img);
            free(resized_img);
            exit(1);
    }

    //stbi_write_bmp("check.bmp", target_w, target_h, channels, (void*)resized_img);

    // prepare input tensor and output tensor
    bm_tensor_t input_tensors[1];
    bmrt_tensor(&input_tensors[0],p_bmrt,net_info->input_dtypes[0],net_info->stages[0].input_shapes[0]);

    bm_tensor_t output_tensors[3];
    for (int i=0;i<3;i++)
        bm_malloc_device_byte(bm_handle, &output_tensors[i].device_mem, net_info->max_output_bytes[i]);

    // prepare input data memory
    float* input_data[1];
    if(is_soc){
        status = bm_mem_mmap_device_mem(bm_handle, &input_tensors[0].device_mem, (void*)&input_data[0]);
        assert(BM_SUCCESS == status);
    } else {
        input_data[0] = (float*)calloc(channels*net_area,sizeof(float));
    }

    // fill the input_data from resized_img
    // input data is CHW, but resized_img is HWC
    float* input_temp0 = input_data[0] + start_y * net_w + start_x;
    unsigned temp_w = target_w*channels;
    for (int k=0;k<channels;k++){
        float* input_temp1 = input_temp0 + k*net_area;
        unsigned char* r_temp1 = resized_img + k;
        for (int i=0;i<target_h;i++){
            float* input_temp2 = input_temp1 + i*net_w;
            unsigned char* r_temp2 = r_temp1 + i*temp_w;
            for (int j=0;j<target_w;j++){
                input_temp2[j] = (float)r_temp2[j*channels]/255.0;
            }
        }
    }
    free(resized_img);

    // flush the cache or s2d
    if(is_soc){
        status = bm_mem_flush_device_mem(bm_handle, &input_tensors[0].device_mem);
    } else {
        status = bm_memcpy_s2d_partial(bm_handle, input_tensors[0].device_mem, (void *)input_data[0], bmrt_tensor_bytesize(&input_tensors[0]));
        free(input_data[0]);
    }
    assert(BM_SUCCESS == status);

    // do inference
    ret = bmrt_launch_tensor_ex(p_bmrt, net_names[0], input_tensors, 1, output_tensors, 3, true, false);
    assert(true == ret);

    // sync, wait for finishing inference
    bm_thread_sync(bm_handle);

    if (is_soc){
        status = bm_mem_unmap_device_mem(bm_handle, input_data[0], bm_mem_get_device_size(input_tensors[0].device_mem));
        assert(BM_SUCCESS == status);
    }

    // prepare output data
    float* output[3];
    if (is_soc){
        for (int i=0;i<3;i++){
            status = bm_mem_mmap_device_mem(bm_handle, &output_tensors[i].device_mem, (void*)&output[i]);
            assert(BM_SUCCESS == status);
            status = bm_mem_invalidate_device_mem(bm_handle, &output_tensors[i].device_mem);
            assert(BM_SUCCESS == status);
        }
    } else {
        for (int i=0;i<3;i++){
            output[i] = (float*)malloc(net_info->max_output_bytes[i]);
            bm_memcpy_d2s_partial(bm_handle, output[i], output_tensors[i].device_mem, bmrt_tensor_bytesize(&output_tensors[i]));
        }
    }

    // postprocess
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
            float *ptr = output[tidx] + anchor_idx*feature_size;
            for (int i = 0; i < area; i++) {
                dst[0] = (sigmoid(ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * net_w;
                dst[1] = (sigmoid(ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * net_h;
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
    if (is_soc){
        for (int i=0;i<3;i++){
            status = bm_mem_unmap_device_mem(bm_handle, output[i], bm_mem_get_device_size(output_tensors[i].device_mem));
            assert(BM_SUCCESS == status);
        }
    } else {
        for (int i=0;i<3;i++){
            free(output[i]);
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

    // get coco names
    FILE *file = fopen("coco.names", "r");
    if (file == NULL) {
        perror("Error opening coco.names");
        return 1;
    }
    int max_lines = 80;
    int max_length = 256;
    // create a pointer to save each line
    char **lines = (char **)malloc(max_lines * sizeof(char *));
    if (lines == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return 1;
    }
    // read each line
    char buffer[max_length];
    int i = 0;
    while (i < max_lines && fgets(buffer, max_length, file) != NULL) {
        // allocate memory for each line and copy the content
        lines[i] = (char *)malloc((strlen(buffer) + 1) * sizeof(char));
        if (lines[i] == NULL) {
            perror("Memory allocation failed for line");
            fclose(file);
            for (int j = 0; j < i; ++j) {
                free(lines[j]);
            }
            free(lines);
            return 1;
        }
        strcpy(lines[i], buffer);
        i++;
    }

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
            box->x = (box->x - start_x) / ratiox;
            box->y = (box->y - start_y) / ratioy;
            box->w = box->w / ratiox;
            box->h = box->h / ratioy;
            fix_box(box,width,height);
            int color_id = box->class_id % colors_num;
            draw_rect(img,box,width,colors[color_id]);
            box_id++;
            printf("class[%02d]: scores = %f, label = %s",box_id,box->score,lines[box->class_id]);
        }
    }
    free(keep);

    // free the memory for coco.names
    for (int j = 0; j < i; ++j) {
        free(lines[j]);
    }
    free(lines);

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
    stbi_write_bmp(result_name, width, height, channels, (void*)img);
    printf("Save result bmp to : %s\n", result_name);

    // free original image memory
    stbi_image_free(img);

    // free result box struct
    free(yolobox);
    // at last, free device memory
    for (int i = 0; i < net_info->input_num; ++i) {
        bm_free_device(bm_handle, input_tensors[i].device_mem);
    }
    for (int i = 0; i < net_info->output_num; ++i) {
        bm_free_device(bm_handle, output_tensors[i].device_mem);
    }
    free(net_names);
    bmrt_destroy(p_bmrt);
    bm_dev_free(bm_handle);

    return 0;
}
