#include <fstream>
#include <string>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "mpi_nnie_ssd.h"
#include "detectionCom.h"

#define SSD_THRSHOLD 0.5f
#define SSD_REPORT_NUM (12)

//extern const HI_CHAR *g_paszPicList_d[];

//extern const HI_CHAR *g_paszModelName_d[];

void getSSDParm(SVP_NNIE_SSD_S *param, const SVP_NNIE_ONE_SEG_DET_S *pstDetecionParam)
{
    for (HI_S32 i = 0; i < 12; i++)
    {
        param->conv_width[i] = pstDetecionParam->astDst[i].unShape.stWhc.u32Height;
        param->conv_height[i] = pstDetecionParam->astDst[i].unShape.stWhc.u32Chn;
        param->conv_channel[i] = pstDetecionParam->astDst[i].unShape.stWhc.u32Width;
    }

    for (HI_S32 i = 0; i < 6; i++)
    {
        param->softmax_input_width[i] = param->conv_channel[i * 2 + 1];
    }

    //----------------- Set PriorBox Parameters ---------------
    for (HI_S32 i = 0; i < 6; i++)
    {
        param->priorbox_layer_width[i] = param->conv_width[i * 2];
        param->priorbox_layer_height[i] = param->conv_height[i * 2];
    }

    param->img_width = pstDetecionParam->astSrc->unShape.stWhc.u32Width;
    param->img_height = pstDetecionParam->astSrc->unShape.stWhc.u32Height;

    param->priorbox_min_size[0] = 30;
    param->priorbox_min_size[1] = 60;
    param->priorbox_min_size[2] = 111;
    param->priorbox_min_size[3] = 162;
    param->priorbox_min_size[4] = 213;
    param->priorbox_min_size[5] = 264;

    param->priorbox_max_size[0] = 60;
    param->priorbox_max_size[1] = 111;
    param->priorbox_max_size[2] = 162;
    param->priorbox_max_size[3] = 213;
    param->priorbox_max_size[4] = 264;
    param->priorbox_max_size[5] = 315;

    param->min_size_num = 1;
    param->max_size_num = 1;
    param->flip = 1;
    param->clip = 0;

    param->input_ar_num[0] = 1;
    param->input_ar_num[1] = 2;
    param->input_ar_num[2] = 2;
    param->input_ar_num[3] = 2;
    param->input_ar_num[4] = 1;
    param->input_ar_num[5] = 1;

    param->priorbox_aspect_ratio[0][0] = 2;
    param->priorbox_aspect_ratio[0][1] = 0;
    param->priorbox_aspect_ratio[1][0] = 2;
    param->priorbox_aspect_ratio[1][1] = 3;
    param->priorbox_aspect_ratio[2][0] = 2;
    param->priorbox_aspect_ratio[2][1] = 3;
    param->priorbox_aspect_ratio[3][0] = 2;
    param->priorbox_aspect_ratio[3][1] = 3;
    param->priorbox_aspect_ratio[4][0] = 2;
    param->priorbox_aspect_ratio[4][1] = 0;
    param->priorbox_aspect_ratio[5][0] = 2;
    param->priorbox_aspect_ratio[5][1] = 0;

    param->priorbox_step_w[0] = 8;
    param->priorbox_step_w[1] = 16;
    param->priorbox_step_w[2] = 32;
    param->priorbox_step_w[3] = 64;
    param->priorbox_step_w[4] = 100;
    param->priorbox_step_w[5] = 300;

    param->priorbox_step_h[0] = 8;
    param->priorbox_step_h[1] = 16;
    param->priorbox_step_h[2] = 32;
    param->priorbox_step_h[3] = 64;
    param->priorbox_step_h[4] = 100;
    param->priorbox_step_h[5] = 300;

    param->offset = 0.5;

    param->priorbox_var[0] = (HI_S32)(0.1 * 4096);
    param->priorbox_var[1] = (HI_S32)(0.1 * 4096);
    param->priorbox_var[2] = (HI_S32)(0.2 * 4096);
    param->priorbox_var[3] = (HI_S32)(0.2 * 4096);

    //------------------ Set Softmax Parameters --------------------
    param->softmax_in_height = 21;
    param->softmax_in_channel[0] = 121296;
    param->softmax_in_channel[1] = 45486;
    param->softmax_in_channel[2] = 12600;
    param->softmax_in_channel[3] = 3150;
    param->softmax_in_channel[4] = 756;
    param->softmax_in_channel[5] = 84;

    param->concat_num = 6;
    param->softmax_out_width = 1;
    param->softmax_out_height = 21;
    param->softmax_out_channel = 8732;

    //---------------- Set DetectionOut Parameters ----------------
    param->num_classes = 21;
    param->top_k = 400;
    param->keep_top_k = 200;
    param->nms_thresh = (HI_S32)(0.3 * 4096);
    param->conf_thresh = 1;

    param->detect_input_channel[0] = 23104;
    param->detect_input_channel[1] = 8664;
    param->detect_input_channel[2] = 2400;
    param->detect_input_channel[3] = 600;
    param->detect_input_channel[4] = 144;
    param->detect_input_channel[5] = 16;

    param->conv_stride[0] = SVP_SAMPLE_ALIGN16(param->conv_channel[1] * sizeof(HI_S32)) / sizeof(HI_S32);
    param->conv_stride[1] = SVP_SAMPLE_ALIGN16(param->conv_channel[3] * sizeof(HI_S32)) / sizeof(HI_S32);
    param->conv_stride[2] = SVP_SAMPLE_ALIGN16(param->conv_channel[5] * sizeof(HI_S32)) / sizeof(HI_S32);
    param->conv_stride[3] = SVP_SAMPLE_ALIGN16(param->conv_channel[7] * sizeof(HI_S32)) / sizeof(HI_S32);
    param->conv_stride[4] = SVP_SAMPLE_ALIGN16(param->conv_channel[9] * sizeof(HI_S32)) / sizeof(HI_S32);
    param->conv_stride[5] = SVP_SAMPLE_ALIGN16(param->conv_channel[11] * sizeof(HI_S32)) / sizeof(HI_S32);
}

void showResult(SVP_NNIE_SSD_S *para, HI_S32* pstDstScore, HI_S32* pstDstBbox,
    HI_S32* pstRoiOutCnt, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo, HI_U32 *pu32BoxNum,
    string& strResultFolderDir, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    HI_U32 ClassNum = para->num_classes;
    HI_U32 i = 0;
    HI_U32 u32MaxRoi = para->top_k;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_U32 u32Index = 0;
    HI_U32 u32BoxNum = 0;

    HI_S32 s32XMin = 0;
    HI_S32 s32YMin = 0;

    HI_S32 s32XMax = 0;
    HI_S32 s32YMax = 0;

    HI_S32* ps32Score     = pstDstScore;
    HI_S32* ps32Bbox      = pstDstBbox;
    HI_S32* ps32RoiOutCnt = pstRoiOutCnt;

    HI_FLOAT fProb;

    string fileName = strResultFolderDir + imgNamePair.first + "_detResult.txt";
    ofstream fout(fileName.c_str());
    if (!fout.good()) {
        printf("%s open failure!", fileName.c_str());
        return;
    }

    PrintBreakLine(HI_TRUE);

    /* detResult start with origin image height and width */
    fout << pstBoxInfo->u32OriImHeight << "  " << pstBoxInfo->u32OriImWidth << endl;
    cout << pstBoxInfo->u32OriImHeight << "  " << pstBoxInfo->u32OriImWidth << endl;

    //printf("class\tconfidence\txmin\tymin\txmax\tymax\n");
    for (i = 1; i < ClassNum; i++)
    {
        u32ScoreBias = i * u32MaxRoi;
        u32BboxBias = i * u32MaxRoi * SVP_WK_COORDI_NUM;
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / SVP_WK_QUANT_BASE < SSD_THRSHOLD)
            continue;

        for (u32Index = 0; u32Index < (HI_U32)ps32RoiOutCnt[i]; u32Index++)
        {
            fProb = (HI_FLOAT)ps32Score[u32ScoreBias + u32Index] / SVP_WK_QUANT_BASE;
            if (fProb < SSD_THRSHOLD) break;
            s32XMin = SVP_SAMPLE_MAX(SVP_SAMPLE_MIN(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 0], (HI_S32)(para->img_width - 1)), 0);
            s32YMin = SVP_SAMPLE_MAX(SVP_SAMPLE_MIN(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 1], (HI_S32)(para->img_height - 1)), 0);
            s32XMax = SVP_SAMPLE_MAX(SVP_SAMPLE_MIN(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 2], (HI_S32)(para->img_width - 1)), 0);
            s32YMax = SVP_SAMPLE_MAX(SVP_SAMPLE_MIN(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 3], (HI_S32)(para->img_height - 1)), 0);

            pstBoxInfo->pstBbox[u32BoxNum].f32ClsScore = fProb;
            pstBoxInfo->pstBbox[u32BoxNum].f32Xmin = (HI_FLOAT)s32XMin;
            pstBoxInfo->pstBbox[u32BoxNum].f32Xmax = (HI_FLOAT)s32XMax;
            pstBoxInfo->pstBbox[u32BoxNum].f32Ymin = (HI_FLOAT)s32YMin;
            pstBoxInfo->pstBbox[u32BoxNum].f32Ymax = (HI_FLOAT)s32YMax;
            pstBoxInfo->pstBbox[u32BoxNum].u32MaxScoreIndex = i;
            u32BoxNum++;

            HI_CHAR resultLine[512];

            snprintf(resultLine, 512, "%s  %4d  %9.8f  %4d  %4d  %4d  %4d\n",
                imgNamePair.first.c_str(),
                i, fProb, s32XMin, s32YMin, s32XMax, s32YMax);

            fout << resultLine;
            cout << resultLine;

            //printf("%4d\t%9.8f\t%4d\t%4d\t%4d\t%4d\n", i, fProb, s32XMin, s32YMin, s32XMax, s32YMax);
        }
    }

    PrintBreakLine(HI_TRUE);
    fout.close();

    *pu32BoxNum = u32BoxNum;
}


void SvpSampleWkSSDGetResult(SVP_NNIE_ONE_SEG_DET_S *pstDetParam,
    HI_VOID *pstParam, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo, HI_U32 *pu32BoxNum,
    string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_S32 size = 0;
    HI_S32* dst_score;
    HI_S32* dst_bbox;
    HI_S32* dst_roicnt;
    HI_S32* assist;
    HI_U32 i;
    SVP_NNIE_SSD_S *pstSSDParam = (SVP_NNIE_SSD_S*)pstParam;

    /*size = param.num_classes * param.keep_top_k * sizeof(HI_S32);
    dst_score = (HI_S32*)malloc(sizeof(HI_S32)*size);
    size = param.num_classes * param.keep_top_k * 4 * sizeof(HI_S32);
    dst_bbox = (HI_S32*)malloc(sizeof(HI_S32)*size);
    size = param.num_classes * sizeof(HI_S32);
    dst_roicnt = (HI_S32*)malloc(sizeof(HI_S32)*size);*/
    ///assit memeory
    size = pstSSDParam->softmax_out_channel * 8;////prior box 8:x,y,w,h,and corresponding var(4)
    size += pstSSDParam->softmax_out_width * pstSSDParam->softmax_out_height * pstSSDParam->softmax_out_channel;////softMaxout
    size = (size + pstSSDParam->softmax_out_channel * 16)*sizeof(HI_S32) + pstSSDParam->softmax_out_channel * sizeof(SVP_SAMPLE_STACK_S);///16:AllDecodeBoxes:4,singleProposal:6,AfterTopK:6
    //assist = (HI_S32*)malloc(size);

    /************************************/
    HI_S32 *ps32ResultMem = pstDetParam->ps32ResultMem;
    dst_score  = ps32ResultMem;
    dst_bbox = ps32ResultMem + pstSSDParam->num_classes * pstSSDParam->keep_top_k;
    dst_roicnt = dst_bbox + pstSSDParam->num_classes * pstSSDParam->keep_top_k * 4;
    assist = dst_roicnt + pstSSDParam->num_classes;

    /***********************************/

    ///deal with multi frame
    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDetParam->astDst[0].u32Num; u32NumIndex++)
    {
        for (i = 0; i < SSD_REPORT_NUM; i++)
        {
            HI_U32 u32FrameStride = (pstDetParam->astDst[i].u32Stride) *
                                    (pstDetParam->astDst[i].unShape.stWhc.u32Chn) *
                                    (pstDetParam->astDst[i].unShape.stWhc.u32Height);

            pstSSDParam->conv_data[i] = (HI_S32*)((HI_U8*)pstDetParam->astDst[i].u64VirAddr + u32NumIndex * u32FrameStride);
        }

        HI_MPI_SVP_NNIE_SSD_Forward(pstSSDParam, pstSSDParam->conv_data, dst_score, dst_bbox, dst_roicnt, assist);

        showResult(pstSSDParam, dst_score, dst_bbox, dst_roicnt, pstBoxInfo, &pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);
    }
}
