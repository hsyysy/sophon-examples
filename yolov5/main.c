#ifdef __ARM_NEON
#define STBI_NEON
#endif

#include <bmruntime_interface.h>
#include "utils.h"

#include "yolov5.h"

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
    for (;dev_id < (unsigned)total_dev;dev_id++){
        status = bm_dev_request(&bm_handle, dev_id);
        assert(BM_SUCCESS == status);

        unsigned p_chipid;
        bm_get_chipid(bm_handle, &p_chipid);
        if (p_chipid == 0x1684){
            //printf("chip = BM1684\n");
            bm_dev_free(bm_handle);
            if (dev_id == (unsigned)total_dev-1){
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

    struct resize_info r_info;
    r_info.ori_w = width;
    r_info.ori_h = height;
    r_info.net_w = net_info->stages[0].input_shapes->dims[3];
    r_info.net_h = net_info->stages[0].input_shapes->dims[2];
    r_info.ratio_x = 1.0;
    r_info.ratio_y = 1.0;
    r_info.start_x = 0;
    r_info.start_y = 0;
    // initialize resized_img
    int net_area = r_info.net_w*r_info.net_h;

    // prepare input tensor and output tensor
    bm_tensor_t input_tensors[1];
    bmrt_tensor(&input_tensors[0],p_bmrt,net_info->input_dtypes[0],net_info->stages[0].input_shapes[0]);

    bm_tensor_t output_tensors[3];
    for (int i=0;i<3;i++)
        bm_malloc_device_byte(bm_handle, &output_tensors[i].device_mem, net_info->max_output_bytes[i]);

    // prepare input data memory
    float* input_data[1];
    if(is_soc){
        status = bm_mem_mmap_device_mem(bm_handle, &input_tensors[0].device_mem,
                (long long unsigned int*)&input_data[0]);
        assert(BM_SUCCESS == status);
    } else {
        input_data[0] = (float*)calloc(channels*net_area,sizeof(float));
    }

    pre_process(img, input_data[0], &r_info);

    // flush the cache or s2d
    if(is_soc){
        status = bm_mem_flush_device_mem(bm_handle, &input_tensors[0].device_mem);
    } else {
        status = bm_memcpy_s2d_partial(bm_handle, input_tensors[0].device_mem,
                (long long unsigned int*)input_data[0], bmrt_tensor_bytesize(&input_tensors[0]));
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
            status = bm_mem_mmap_device_mem(bm_handle, &output_tensors[i].device_mem,
                    (long long unsigned int*)&output[i]);
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

    post_process(output, img_path, img, &r_info);
    stbi_image_free(img);

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
