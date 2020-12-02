#include <stdio.h>
#include "SvpSampleWk.h"

int main(int argc, char* argv[])
{
    /*set stderr &stdout buffer to NULL to flush print info immediately*/
    setbuf(stderr, NULL);
    setbuf(stdout, NULL);

    /*Classificacion*/
    SvpSampleCnnClfLenet();
    SvpSampleCnnClfAlexnet();
    SvpSampleCnnClfVgg16();
    SvpSampleCnnClfGooglenet();
    SvpSampleCnnClfResnet50();
    SvpSampleCnnClfSqueezenet();

    /*Detection*/
    SvpSampleRoiDetFasterRCNNAlexnet();
    SvpSampleRoiDetFasterRCNNVGG16();
    SvpSampleRoiDetFasterRCNNResnet18();
    SvpSampleRoiDetFasterRCNNResnet34();
    SvpSampleRoiDetFasterRCNNPvanet();
    SvpSampleRoiDetFasterRCNNDoubleRoi();
    SvpSampleRoiDetRFCNResnet50();
    SvpSampleCnnDetYoloV1();
    SvpSampleCnnDetYoloV2();
    SvpSampleCnnDetSSD();

    /*Segmentation*/
    SvpSampleCnnFcnSegnet();

    /*LSTM*/
    SvpSampleRecurrentLSTMFC();
    SvpSampleRecurrentLSTMRelu();

    //printf("press any key to exit ... \n");
    //getchar();

    return 0;
}
