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
int argmax(float* data, int num){
    float max_value = 0.0;
    int max_index = 0;
    for(int i = 0; i < num; ++i) {
        float value = data[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
  }

  return max_index;
}

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
