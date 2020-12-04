#include "detectionCom.h"
#include "ssd_software.h"

using namespace std;

HI_S32 SVP_GetPriorBoxSize(const SVP_NNIE_SSD_S *para, HI_S32 s32Index)
{
    HI_S32 s32ARNum = para->flip == 1 ? para->input_ar_num[s32Index] * 2 + 1 : para->input_ar_num[s32Index] + 1;
    HI_S32 s32PriorNum = para->min_size_num * s32ARNum + para->max_size_num;
    HI_S32 s32CellNum = para->priorbox_layer_width[s32Index] * para->priorbox_layer_height[s32Index];
    return s32CellNum * (8 + (s32ARNum - 1) * 4 + s32PriorNum * 4) * sizeof(HI_S32);
}

HI_S32 SVP_getDetectInputNum(const SVP_NNIE_SSD_S *para)
{
    HI_S32 s32PriorSum = 0;
    for (HI_S32 i = 0; i < para->concat_num; i++)
    {
        s32PriorSum += para->detect_input_channel[i] / 4;
    }
    return s32PriorSum;
}

HI_S32 SVP_GetDetectionOutSize(const SVP_NNIE_SSD_S *para)
{
    HI_S32 s32PriorNum = SVP_getDetectInputNum(para);

    return s32PriorNum*(4 + 6 + 6)*sizeof(HI_S32) + MAX_STACK_DEPTH * sizeof(NNIE_STACK_S);
}

HI_S32 HI_MPI_SVP_NNIE_SSD_Forward(
    SVP_NNIE_SSD_S *para,
    //----- input data
    HI_S32** input_permute_data,
    //----- output data
    HI_S32* dst_score,
    HI_S32* dst_bbox,
    HI_S32* dst_roicnt,
    //----- assist mempool
    HI_S32* assist_mem
)
{
    //------------------------- PriorBox ----------------------------
    // assign assist mem
    HI_S32 size = 0;
    para->priorbox_output_data[0] = assist_mem;

    for (HI_S32 i = 0; i < SVP_NNIE_LAYER_PRIORBOX_CNT; i++)
    {
        size = SVP_GetPriorBoxSize(para, i)/sizeof(HI_S32); // size unit is sizeof(HI_S32)
        para->priorbox_output_data[i + 1] = para->priorbox_output_data[i] + size; // get each priorbox layer output Addr
    }

    for (HI_S32 p = 0; p < SVP_NNIE_LAYER_PRIORBOX_CNT; p++)
    {
        PriorBoxForward(
            para->priorbox_layer_width[p],
            para->priorbox_layer_height[p],
            para->img_width,
            para->img_height,
            para->priorbox_min_size[p],
            para->min_size_num,
            para->priorbox_max_size[p],
            para->max_size_num,
            para->flip,
            para->clip,
            para->input_ar_num[p],
            para->priorbox_aspect_ratio[p],
            para->priorbox_step_w[p],
            para->priorbox_step_h[p],
            para->offset,
            para->priorbox_var,
            para->priorbox_output_data[p]
        );
    }

    //------------------------ Softmax ---------------------------------
    // assign softmax output Addr
    size = para->softmax_out_width * para->softmax_out_height * para->softmax_out_channel;
    para->softmax_output_data = para->priorbox_output_data[6];

    para->softmax_input_data[0] = input_permute_data[1];
    para->softmax_input_data[1] = input_permute_data[3];
    para->softmax_input_data[2] = input_permute_data[5];
    para->softmax_input_data[3] = input_permute_data[7];
    para->softmax_input_data[4] = input_permute_data[9];
    para->softmax_input_data[5] = input_permute_data[11];

    SoftmaxForward(
        para->softmax_in_height,
        para->softmax_in_channel,
        para->concat_num,
        para->conv_stride,
        para->softmax_out_width,
        para->softmax_out_height,
        para->softmax_out_channel,

        para->softmax_input_data,
        para->softmax_output_data
    );

    //----------------------- detection out -----------------------------
    para->detection_out_loc_data[0] = input_permute_data[0];
    para->detection_out_loc_data[1] = input_permute_data[2];
    para->detection_out_loc_data[2] = input_permute_data[4];
    para->detection_out_loc_data[3] = input_permute_data[6];
    para->detection_out_loc_data[4] = input_permute_data[8];
    para->detection_out_loc_data[5] = input_permute_data[10];

    para->detection_out_assist_mem = para->softmax_output_data + size;

    DetectionOutForward(
        para->concat_num,
        para->conf_thresh,
        para->num_classes,
        para->top_k,
        para->keep_top_k,
        para->nms_thresh,
        para->detect_input_channel,

        para->detection_out_loc_data,
        para->priorbox_output_data,
        para->softmax_output_data,
        para->detection_out_assist_mem,
        dst_score,
        dst_bbox,
        dst_roicnt
    );

    return HI_SUCCESS;
}
