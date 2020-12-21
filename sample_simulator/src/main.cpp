#include <stdio.h>
#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "hi_nnie.h"

int main(int argc, char* argv[])
{
    /*set stderr &stdout buffer to NULL to flush print info immediately*/
    setbuf(stderr, NULL);
    setbuf(stdout, NULL);

  SvpSampleCnnDetYoloV3();

    /*Segmentation*/
//    SvpSampleCnnFcnSegnet();


//    //printf("press any key to exit ... \n");
    //getchar();

    return 0;
}
