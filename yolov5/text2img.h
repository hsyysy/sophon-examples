#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
#endif

#include <stdio.h>

struct image {
    int w;
    int h;
    unsigned char rgb[];
};

struct image* image_create(int w, int h) {
    struct image *m = malloc(2*sizeof(int) + (size_t)3 * w * h);
    m->w = w;
    m->h = h;
    return m;
}

struct image* image_load(FILE *in) {
    char c;
    int w, h;
    struct image *m = 0;
    if (fscanf(in, "P6 %d %d 255%c", &w, &h, &c) == 3 && c == '\n') {
        m = image_create(w, h);
        unsigned num = fread(m->rgb, h * 3, w, in);
        if (num != w) {
            fprintf(stderr, "Error: fread failed to read expected number of elements\n");
            exit(1);
        }
    }
    return m;
}

static void draw_char(struct image *imag, int i, int c, const struct image *font) {
    if (c < ' ' || c > '~')
        c = ' ';
    int fx = c % 16;
    int fy = c / 16 - 2;
    int fw = font->w / 16;
    int fh = font->h / 6;
    int offset1 = 3 * imag->w;
    int offset2 = 3 * font->w;
    size_t char_size = 3 * fw;
          unsigned char* dst = imag->rgb + char_size * i;
    const unsigned char* src = font->rgb + 3 * ( font->w * fy*fh + fx*fw );
    for (int y = 0; y < fh; y++) {
        memcpy(dst + offset1 * y, src + offset2 * y, char_size);
    }
}

struct image* get_textimg(const char* font_file, const char* text){
    FILE *fontfile = fopen(font_file, "rb");
    struct image *font = image_load(fontfile);
    //printf("single char: w = %d, h = %d\n",font->w/16,font->h/6);
    fclose(fontfile);
    size_t len = strlen(text);

    struct image* image = image_create(font->w/16 * len, font->h/6);
    for (size_t i = 0; i < len; i++) {
        draw_char(image, i, text[i], font);
    }

    free(font);
    return image;
}

void put_text(unsigned char* img, int width, int height, const char* text, int pos_x, int pos_y, float r){
    const char* font_file = "font32.ppm";

    struct image* image = get_textimg(font_file, text);

    int new_w = (int)image->w*r;
    int new_h = (int)image->h*r;

    if (pos_x + new_w > width){
        new_w = width - pos_x;
        new_h = new_w * (float)image->h / image->w;
    }

    pos_y = pos_y - new_h;
    if (pos_y < 0) pos_y = 0;

    unsigned char *resized_img = (unsigned char *)malloc(new_w * new_h * 3);
    stbir_resize_uint8_linear(image->rgb, image->w, image->h, 0, resized_img, new_w, new_h, 0, STBIR_RGB);

    //stbi_write_bmp("text.bmp", new_w, new_h, 3, (void*)resized_img);

    for (int i=0;i<new_h;i++){
        memcpy(img+3*((i+pos_y)*width+pos_x), resized_img + 3*i*new_w,3*new_w);
        /*
        for (int j=0;j<new_w;j++){
            float r = (float)resized_img[3*(i*new_w + j)];
            float g = (float)resized_img[3*(i*new_w + j) + 1];
            float b = (float)resized_img[3*(i*new_w + j) + 2];
            if ((r+g+b)<300){
                unsigned char* or = img + 3*((i+pos_y)*width+pos_x+j);
                unsigned char* og = or + 1;
                unsigned char* ob = og + 1;
                *or = *or*0.2 + r*0.8;
                *og = *og*0.2 + g*0.8;
                *ob = *ob*0.2 + b*0.8;
            }
        }
        */
    }

    free(resized_img);
    free(image);
}
