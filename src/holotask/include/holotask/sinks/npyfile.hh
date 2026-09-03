#pragma once

#include <string>

#include <vector>
#include <sstream>

namespace npyfile {

    struct Header {
        const std::vector<int>& shape;
        const std::string& dtype;
        int frame_count;
        uint16_t bits_per_pixel;
        uint32_t frame_width;
        uint32_t frame_height;
        uint32_t frame_count;
    };


    class Writer {
        public:
        explicit Writer(const std::string &path, const Header& header);
        ~Writer();

        private:
        FILE* file_;
    };
} 

namespace holotask::sinks {
    
}