#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "mpi_nnie.h"

#define LSTM_UT_EXPOSE_HID 2
#define LSTM_UT_WITH_STATIC 1
#define LSTM_UT_HDR_MAGIC 0xbebabeba


const static HI_CHAR *g_paszPicList_c[] =
{
    "../data/lstm/sentences_list.txt",
    "../data/lstm/sentences_list.txt",
};

#ifndef USE_FUNC_SIM /* inst wk */
const static HI_CHAR *g_paszModelName_c[] = {
    "../data/lstm/lstm_fc/inst/inst_lstm_fc_inst.wk",
    "../data/lstm/lstm_relu/inst/inst_lstm_relu_inst.wk",
};
#else /* func wk */
const static HI_CHAR *g_paszModelName_c[] = {
    "../data/lstm/lstm_fc/inst/inst_lstm_fc_func.wk",
    "../data/lstm/lstm_relu/inst/inst_lstm_relu_func.wk",
};
#endif

struct SentenceDesc
{
    HI_U8 u8Flags;
    HI_U32 u32FrameNum;
    HI_U32 u32Offset;
};

struct LSTMFileHdr
{
    HI_U32 u32Magic;
    HI_U32 u32IVecLen;
    HI_U32 u32OVecLen;
    HI_U8 u32ElementSize;
    HI_U32 u32SentenceNum;
    struct SentenceDesc astDescs[0];
};

static HI_U32 SvpSampleSaveLSTMVector(const SVP_BLOB_S *astDst, FILE *fp, SVP_SAMPLE_LSTMRunTimeCtx *pstCtx)
{
    HI_U32 u32Size = astDst->u32Stride;
    HI_U8 *img = (HI_U8*)astDst->u64VirAddr;
    for (HI_U32 n = 0; n < pstCtx->u32SeqNr; ++n)
    {
        HI_U32 u32FrameNum = pstCtx->pu32Seqs[n];
        if (astDst->enType == SVP_BLOB_TYPE_VEC_S32)
        {
            // h0/c0 report node only contain one vector per sentence.
            fwrite(img, 1, u32Size, fp);
            img += u32Size;
        }
        else // SVP_BLOB_TYPE_SEQ_S32
        {
            for (HI_U32 t = 0; t < u32FrameNum; ++t)
            {
                fwrite(img, 1, u32Size, fp);
                img += u32Size;
            }
        }
    }

    return HI_SUCCESS;
}

static HI_U32 SvpSampleSaveOutput(const SVP_BLOB_S *astDst, SVP_SAMPLE_LSTMRunTimeCtx *pstCtx)
{
    HI_U32 res = HI_FAILURE;

    FILE *fp = fopen("lstm_sample.bin", "wb+");
    for (HI_U32 node_idx = 0; node_idx < SVP_NNIE_MAX_OUTPUT_NUM; ++node_idx)
    {
        if (astDst[node_idx].u64PhyAddr == 0) goto exit;

        res = SvpSampleSaveLSTMVector(&astDst[node_idx], fp, pstCtx);

        if (res) goto exit;
    }

exit:
    SvpSampleCloseFile(fp);
    return res;
}

static void SvpSampleDoReadRandomFile(const char *filename, HI_U8* buf, HI_U32 vecNr, HI_U32 vecSize, HI_U32 vecStride)
{
    FILE *fp = SvpSampleOpenFile(filename, "rb");

    for (HI_U32 i = 0; i < vecNr; ++i)
    {
        if (1 != fread(buf, vecSize, 1, fp))
        {
            printf("fread error");
            SvpSampleCloseFile(fp);
            return;
        }
        buf += vecStride;
    }

    SvpSampleCloseFile(fp);
}

static HI_U32 SvpSampleReadRandomFile(const char* basename, HI_U32 SentenceNr, HI_U32 vecNr, HI_U8* buf, HI_U32 vecSize)
{
    HI_U32 iterSz = 0;
    char filename[SVP_SAMPLE_MAX_PATH];

    if (SentenceNr > 6) return HI_FAILURE;
    FILE *sentenceListFile = fopen(basename, "r");
    for (HI_U32 i = 0; i < SentenceNr; ++i, buf += iterSz)
    {
        iterSz = (vecNr + i) * SVP_SAMPLE_ALIGN16(vecSize);
        if (NULL == fgets(filename, SVP_SAMPLE_MAX_PATH, sentenceListFile))
        {
            printf("read file error");
            fclose(sentenceListFile);
            return HI_FAILURE;
        }
        size_t filenameLen = strlen(filename) - 1;
        if (*filename && filename[filenameLen] == '\n')
        {
            filename[filenameLen] = '\0';
        }
        SvpSampleDoReadRandomFile(filename, buf, vecNr + i, vecSize, SVP_SAMPLE_ALIGN16(vecSize));
    }
    fclose(sentenceListFile);
    return HI_SUCCESS;
}

static HI_U32 SvpSampleReadRandomFileSt(const char* basename, HI_U32 SentenceNr, HI_U8* buf, HI_U32 vecSize)
{
    HI_U32 iterSz = 0;
    char filename[SVP_SAMPLE_MAX_PATH];

    if (SentenceNr > 6) return HI_FAILURE;

    FILE *sentenceListFile = fopen(basename, "r");
    for (HI_U32 i = 0; i < SentenceNr; ++i, buf += iterSz)
    {
        iterSz = SVP_SAMPLE_ALIGN16(vecSize);
        if (NULL == fgets(filename, SVP_SAMPLE_MAX_PATH, sentenceListFile))
        {
            printf("read file error");
            fclose(sentenceListFile);
            return HI_FAILURE;
        }
        size_t filenameLen = strlen(filename) - 1;
        if (*filename && filename[filenameLen] == '\n')
        {
            filename[filenameLen] = '\0';
        }
        SvpSampleDoReadRandomFile(filename, buf, 1, vecSize, SVP_SAMPLE_ALIGN16(vecSize));
    }
    fclose(sentenceListFile);
    return HI_SUCCESS;
}
// this code is for testing not for commit
static void SvpSampleReadRandomFile(SVP_NNIE_CFG_S *pstWkCfg, SVP_NNIE_ONE_SEG_S *pstCommonParam, SVP_SAMPLE_LSTMRunTimeCtx *pstCtx)
{
    HI_U8 nodeIdx = 0;
    HI_U8 *xt = (HI_U8*)pstCommonParam->astSrc[nodeIdx].u64VirAddr;
    HI_U8 *xst = NULL, *h0 = NULL, *c0 = NULL;
    HI_U32 vecSize = pstCommonParam->astSrc[nodeIdx++].unShape.stSeq.u32Dim;

    SvpSampleReadRandomFile(pstWkCfg->paszPicList[0], pstCtx->u32SeqNr, pstCtx->pu32Seqs[0], xt, vecSize * 4);

    if (pstCtx->u8WithStatic)
    {
        xst = (HI_U8*)pstCommonParam->astSrc[nodeIdx].u64VirAddr;
        vecSize = pstCommonParam->astSrc[nodeIdx++].unShape.stSeq.u32Dim;
        SvpSampleReadRandomFileSt(pstWkCfg->paszPicList[0], pstCtx->u32SeqNr, xst, vecSize * 4);
    }

    if (pstCtx->u8ExposeHid)
    {
        h0 = (HI_U8*)pstCommonParam->astSrc[nodeIdx].u64VirAddr;
        vecSize = pstCommonParam->astSrc[nodeIdx++].unShape.stSeq.u32Dim;
        SvpSampleReadRandomFileSt(pstWkCfg->paszPicList[0], pstCtx->u32SeqNr, h0, vecSize * 4);

        c0 = (HI_U8*)pstCommonParam->astSrc[nodeIdx].u64VirAddr;
        vecSize = pstCommonParam->astSrc[nodeIdx++].unShape.stSeq.u32Dim;
        SvpSampleReadRandomFileSt(pstWkCfg->paszPicList[0], pstCtx->u32SeqNr, c0, vecSize * 4);
    }
}

void SvpSampleCreateLSTMCtx(SVP_SAMPLE_LSTMRunTimeCtx *pstCtx, HI_U32 u32SentenceNr, HI_U32 u32BaseFrameNr,
    HI_U8 u8ExposeHid, HI_U8 u8WithStatic)
{
    HI_U32 *seqs = (HI_U32*)malloc(u32SentenceNr * sizeof(HI_U32));
    HI_U32 totalT = 0, maxT = 0;

    for (HI_U32 i = 0; i < u32SentenceNr; ++i)
    {
        seqs[i] = u32BaseFrameNr + i;
        totalT += seqs[i];
        if (seqs[i] > maxT)
            maxT = seqs[i];
    }

    SVP_SAMPLE_LSTMRunTimeCtx ctx =
    {
        seqs,
        u32SentenceNr,
        maxT,
        totalT,
        u8ExposeHid,
        u8WithStatic,
    };

    *pstCtx = ctx;
}

void SvpSampleDestoryLSTMCtx(SVP_SAMPLE_LSTMRunTimeCtx *pstCtx)
{
    if (pstCtx)
        free(pstCtx->pu32Seqs);
}

HI_BOOL SvpSampleWkLSTM(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], HI_U32 u32PicListNum,
    HI_U32 *pu32SrcAlign, HI_U32 *pu32DstAlign, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{

    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32MaxInputNum = 10;
    SVP_NNIE_HANDLE SvpNnieHandle;
    HI_BOOL bInstant = HI_TRUE;
    HI_U32  u32TopN = 5;

    SVP_NNIE_ONE_SEG_S stClfParam = { 0 };
    SVP_NNIE_CFG_S       stClfCfg;
    stClfCfg.pszModelName = pszModelName;
    memcpy(&stClfCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*u32PicListNum);
    stClfCfg.u32MaxInputNum = u32MaxInputNum;
    stClfCfg.u32TopN = u32TopN;
    stClfCfg.u32MaxBboxNum = 0;
    stClfCfg.bNeedLabel = HI_FALSE;

    s32Ret = SvpSampleLSTMInit(&stClfCfg, &stClfParam, pstLSTMCtx);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): WkInit failed!", s32Ret);

//the following code to be modify
    // fill the missing LSTM sentences
    SvpSampleReadRandomFile(&stClfCfg, &stClfParam, pstLSTMCtx);

    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, stClfParam.astSrc, &stClfParam.stModel,
        stClfParam.astDst, &stClfParam.stCtrl, bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): CNN_Forward failed!", s32Ret);

    // for debugging
    SvpSampleSaveOutput(stClfParam.astDst, pstLSTMCtx);

    SvpSampleLSTMDeinit(&stClfParam);

    return HI_TRUE;

Fail:
    SvpSampleOneSegCnnDeinit(&stClfParam);
    return HI_FALSE;
}


void SvpSampleRecurrentLSTMFC()
{
    printf("%s start ...\n", __FUNCTION__);
    // N = 6, T1 = 17, T2 = 18, T3 = 19, T4 = 20, T5 = 21, T6 = 22 ...
    const HI_U32 SentenceNR = 1;
    const HI_U32 INPUT_VEC_NR = 3;
    SVP_SAMPLE_LSTMRunTimeCtx ctx;

    SvpSampleCreateLSTMCtx(&ctx, SentenceNR, INPUT_VEC_NR, !LSTM_UT_EXPOSE_HID, LSTM_UT_WITH_STATIC);

    SvpSampleWkLSTM(g_paszModelName_c[0], g_paszPicList_c, 2, 0, 0, &ctx);

    SvpSampleDestoryLSTMCtx(&ctx);

    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
void SvpSampleRecurrentLSTMRelu()
{
    printf("%s start ...\n", __FUNCTION__);
    // N = 6, T1 = 17, T2 = 18, T3 = 19, T4 = 20, T5 = 21, T6 = 22 ...
    const HI_U32 SentenceNR = 1;
    const HI_U32 INPUT_VEC_NR = 3;
    SVP_SAMPLE_LSTMRunTimeCtx ctx;

    SvpSampleCreateLSTMCtx(&ctx, SentenceNR, INPUT_VEC_NR, !LSTM_UT_EXPOSE_HID, LSTM_UT_WITH_STATIC);

    SvpSampleWkLSTM(g_paszModelName_c[1], g_paszPicList_c, 2, 0, 0, &ctx);

    SvpSampleDestoryLSTMCtx(&ctx);

    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
