#include "holotask/sinks/npyfile.hh"

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>


namespace npyfile {
    Writer::Writer(const std::string &path, const Header& header) {
        file_ = fopen(path.c_str(), "wb");
        if (!file_) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }

        fwrite("\x93NUMPY", 1, 6, file_);
        const unsigned char version[] = {1, 0};
        fwrite(version, 1, sizeof(version), file_);
        
        std::stringstream header_stream;
        header_stream << "{'descr': '" << header.dtype << "', 'fortran_order': False, 'shape': (";
        for (size_t i = 0; i < header.shape.size(); ++i) {
            header_stream << header.shape[i];
            if (i < header.shape.size() - 1) {
                header_stream << ", ";
            }
        }
        header_stream << "), }";
        const std::string header_str = header_stream.str();
        const uint16_t header_len = static_cast<uint16_t>(header_str.size());
        fwrite(&header_len, sizeof(header_len), 1, file_);
        fwrite(header_str.data(), 1, header_str.size(), file_);
    }

    Writer::~Writer() {
        if (file_) {
            fclose(file_);
        }
    }
}