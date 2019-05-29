#include "w2xconv.h"

#if defined(WIN32) && defined(UNICODE)
int wmain(void)
#else
int main(int argc, char** argv)
#endif
{
#if defined(WIN32) && defined(UNICODE)
	int argc = 0;
	LPWSTR *argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    LPWSTR models = L"models_rgb";
#else
    const char *models = "models_rgb";
#endif
    struct W2XConv *c = w2xconv_init(1, 0, 1, 0);
    if (argc >= 2) {
        models = argv[1];
    }
    w2xconv_load_models(c, models);
    w2xconv_test(c, 0);
    w2xconv_fini(c);

    return 0;
}
