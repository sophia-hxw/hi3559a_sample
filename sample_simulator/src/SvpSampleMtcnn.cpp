/*
* Copyright @ Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
* Description: MTCNN sample
* Notes: 1. MTCNN preprocessing needs to resize the image to different scales, and then get all the bounding boxes through P-NET.
         Since P-NET is a convolutional network, it can support input images of any scale. But wk compilation requires a fixed
         input scale, so this sample takes only one valid scale and uses a fixed image size and scale in s_scales.
         2. MTCNN converts the input image to different scales in order to match the face in the image to the frame size of 12*12,
         or most of the face can fall into the 12*12 box. Training related. For example, the size of the face in the picture in
         this example is about 1/5 of the total size, 450 * 1/5 * 0.14 is approximately equal to 12, so 0.14 is an effective scale.
*/

#include "SvpSampleCom.h"
#include "SvpSampleMtcnn.h"
#include "mtcnn_interface.h"

/* The first two values represent the height and width of the input image.
The third represents the scale of resize, which needs to be changed when the input image changes.
Currently writing to the size and scale of the fixed image */
static HI_DOUBLE s_scales[] = { 344, 450, 0.14189182274208526 };

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_mtcnnModelName[] = {
    "../../data/detection/mtcnn/inst/12net_mtcnn_inst.wk",
    "../../data/detection/mtcnn/inst/24net_mtcnn_inst.wk",
    "../../data/detection/mtcnn/inst/48net_mtcnn_inst.wk",
    nullptr
};
#else /* func wk */
const HI_CHAR *g_mtcnnModelName[] = {
    "../../data/detection/mtcnn/inst/12net_mtcnn_func.wk",
    "../../data/detection/mtcnn/inst/24net_mtcnn_func.wk",
    "../../data/detection/mtcnn/inst/48net_mtcnn_func.wk",
    nullptr
};
#endif

const HI_CHAR *g_image = "./../../data/detection/images/test/two_persons_face.jpg";
const HI_CHAR* g_mtcnnResultPath = "result_SVP_SAMPLE_MTCNN/";

void SvpSampleMtcnn()
{
    const HI_U32 pNetModelNum = 1;
    std::vector<HI_CHAR*> models;
    HI_U32 i = 0;
    while (g_mtcnnModelName[i] != nullptr) {
        models.push_back(const_cast<HI_CHAR*>(g_mtcnnModelName[i]));
        i++;
    }

    CHECK_EXP_VOID_RET(sizeof(s_scales) / sizeof(HI_DOUBLE) != pNetModelNum + 2, "P-NET model number is not expect value[%d]", pNetModelNum);
    std::vector<HI_FLOAT> scales;
    i = 0;
    while (i < pNetModelNum + 2) {
        scales.push_back((HI_FLOAT)s_scales[i]);
        i++;
    }

    (void)SvpSampleCnnDetectionMtcnn(models, pNetModelNum, g_image, scales);
}

void SvpSampleCnnDetMtcnn()
{
    printf("%s start ...\n", __FUNCTION__);

    SvpSampleMtcnn();
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}