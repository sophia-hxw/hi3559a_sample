#ifndef _GEN_ROI_INFO_H_
#define _GEN_ROI_INFO_H_


#include <stdlib.h>
#include <stdio.h>
#include "malloc.h"
#include "math.h"
#include "time.h"
#include "hi_nnie.h"

typedef unsigned int  t_uint;
typedef int  t_int;
typedef unsigned char t_uchar;

typedef struct ROI_INFO
{
    t_uint input_channel;
    t_uint input_height;
    t_uint input_width;
    t_uint is_ping_pong_mode;
    t_uint total_is_ping_pong_mode;
    t_uint box_number;
    t_uint box_height;
    t_uint box_width;
    t_uint is_calc_high_precision;
    t_uint scale_20_12;
    t_uint roi_flag;

    t_uint reshape_input_channel;
    t_uint reshape_input_width;
    t_uint din_w_ram_offset;
    t_uint din_h_ram_offset;
    t_uint total_din_w_ram_offset;
    t_uint total_din_h_ram_offset;
    t_uint din_block_height_max;
    t_uint din_block_height_half;
    t_uint block_num;

    t_uint *info_head_addr;
    t_uint *kernel_num_addr;
    t_uint *blk_info_addr;
    t_uint *org_index_addr;
}t_ROI_INFO;

t_uint ceil_div(t_uint x, t_uint y);

t_int roi_info_initial(
    t_uint input_channel,
    t_uint input_height,
    t_uint input_width,
    t_uint is_ping_pong_mode,
    t_uint total_is_ping_pong_mode,
    t_uint box_number,
    t_uint box_height,
    t_uint box_width,
    t_uint is_calc_high_precision,
    t_uint roi_flag,
    t_uint din_block_height_max,
    t_uint block_num,
    t_uint din_w_ram_offset,
    t_uint din_h_ram_offset,
    t_uint total_din_w_ram_offset,
    t_uint total_din_h_ram_offset,
    t_uint scale,
    t_ROI_INFO *roi_info);

t_uint  roi_interface(t_uint *left_x, t_uint *right_x, t_uint *top_y, t_uint *bottom_y, t_ROI_INFO *roi_info, t_uint *indicator, t_uint *ddr_addr);
t_uint  psroi_interface(t_uint *left_x, t_uint *right_x, t_uint *top_y, t_uint *bottom_y, t_ROI_INFO *roi_info, t_uint *indicator, t_uint *ddr_addr);

#endif /*_GEN_ROI_INFO_H_*/


