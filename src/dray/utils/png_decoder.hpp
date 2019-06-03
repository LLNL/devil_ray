#ifndef DRAY_PNG_DECODER_HPP
#define DRAY_PNG_DECODER_HPP

#include <string>

namespace dray
{

class PNGDecoder
{
public:
    PNGDecoder();
    ~PNGDecoder();
    // rgba
    void decode(unsigned char *&rgba,
                int &width,
                int &height,
                const std::string &file_name);
};

};

#endif


