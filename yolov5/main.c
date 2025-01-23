#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <bmruntime_interface.h>
#include <sys/stat.h>
#include "utils.h"

int main(int argc, char** argv){
    int total_dev;
    bm_dev_getcount(&total_dev);
    //printf("Total devices num = %d\n",total_dev);

    // request bm_handle
    bm_handle_t bm_handle;
    bm_status_t status;
    unsigned dev_id = 0;
#if defined(__arm__) || defined(__aarch64__)
    status = bm_dev_request(&bm_handle, dev_id);
    assert(BM_SUCCESS == status);
#else
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

    // 初始化resized_img
    int net_h = net_info->stages[0].input_shapes->dims[2];
    int net_w = net_info->stages[0].input_shapes->dims[3];

    // prepare resized_img memory
    unsigned char *resized_img;
    bm_device_mem_t resized_img_dev;
    if (is_soc){
        bm_malloc_device_byte(bm_handle, &resized_img_dev, net_w * net_h * channels);
        status = bm_mem_mmap_device_mem(bm_handle, &resized_img_dev, (void*)&resized_img);
        assert(BM_SUCCESS == status);
    } else {
        resized_img = (unsigned char *)malloc(net_w * net_h * channels);
        if (resized_img == NULL) {
            printf("Unable to allocate memory for the resized image.\n");
            stbi_image_free(img);
            exit(1);
        }
    }

    // using stb_image_resize to resize
    if (!stbir_resize_uint8_linear(img, width, height, 0, resized_img, net_w, net_h, 0, STBIR_RGB)) {
            printf("Failed to resize image\n");
            stbi_image_free(img);
            free(resized_img);
            exit(1);
    }
    if (is_soc){
        status = bm_mem_flush_device_mem(bm_handle, &resized_img_dev);
        assert(BM_SUCCESS == status);
    }

    //int result = stbi_write_bmp("check.bmp", net_w, net_h, channels, (void*)resized_img);

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
        input_data[0] = (float*)malloc(channels*net_w*net_h*sizeof(float));
    }
    // fill the input_data from resized_img
    for (int k=0;k<channels;k++){
        unsigned channel_id = k*net_h*net_w;
        for (int i=0;i<net_h;i++){
            unsigned w_id = i*net_w;
            unsigned wh_id = channel_id + w_id;
            for (int j=0;j<net_w;j++){
                input_data[0][wh_id + j] = (float)resized_img[(w_id+j)*channels + k]/255.0;
            }
        }
    }
    // link the input_data with the input tensor
    if(is_soc){
        // resized_img
        status = bm_mem_unmap_device_mem(bm_handle, resized_img, bm_mem_get_device_size(resized_img_dev));
        assert(BM_SUCCESS == status);
        bm_free_device(bm_handle, resized_img_dev);
        // input_tensor
        status = bm_mem_flush_device_mem(bm_handle, &input_tensors[0].device_mem);
        assert(BM_SUCCESS == status);
    } else {
        // resized_img
        free(resized_img);
        // input_tensor
        bm_memcpy_s2d_partial(bm_handle, input_tensors[0].device_mem, (void *)input_data[0], bmrt_tensor_bytesize(&input_tensors[0]));
    }

    // do inference
    ret = bmrt_launch_tensor_ex(p_bmrt, net_names[0], input_tensors, 1, output_tensors, 3, true, false);
    assert(true == ret);

    // sync, wait for finishing inference
    bm_thread_sync(bm_handle);

    if (is_soc){
        status = bm_mem_unmap_device_mem(bm_handle, input_data[0], bm_mem_get_device_size(input_tensors[0].device_mem));
        assert(BM_SUCCESS == status);
    } else {
        free(input_data[0]);
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
    unsigned max_wh = 7680;
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
            if (confidence * score > m_confThreshold) {
                struct YoloV5Box* box = &yolobox[box_i];
                unsigned c = class_id * max_wh;
                float w = ptr[2];
                float h = ptr[3];
                box->x        = ptr[0] - w / 2 + c;
                box->y        = ptr[1] - h / 2 + c;
                box->width    = w;
                box->height   = h;
                box->class_id = class_id;
                box->score    = confidence * score;

                if (box->x < 0) box->x = 0;
                if (box->y < 0) box->y = 0;
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
    float ratiox = (float)net_w/width;
    float ratioy = (float)net_h/height;
    float nmsConfidence = 0.6;
    int tx1 = 0;
    int ty1 = 0;
    bool* keep = (bool*)malloc(box_i*sizeof(bool));
    memset(keep, true, box_i*sizeof(bool));
    NMS(yolobox, keep, nmsConfidence, box_i);
    // plot the rect on the img
    for (int i=0;i<box_i;i++){
        if (keep[i]){
            struct YoloV5Box* box = &yolobox[i];
            unsigned c = box->class_id * max_wh;
            box->x  = (box->x - tx1 - c) / ratiox;
            box->y  = (box->y - ty1 - c) / ratioy;
            box->width  = (box->width) / ratiox;
            box->height = (box->height) / ratioy;
            draw_rect(img,box,width,height,colors[i]);
            printf("class[%02d]: scores = %f, label = %s",i,box->score,lines[box->class_id]);
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
    bmrt_destroy(p_bmrt);
    bm_dev_free(bm_handle);

    return 0;
}
