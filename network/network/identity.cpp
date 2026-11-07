#include "identity.hpp"

#include <mbedtls/ecp.h>
#include <mbedtls/sha256.h>
#include <mbedtls/x509.h>
#include <mbedtls/x509_crt.h>
#include <mbedtls/pk.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace net::secure::identity {
namespace {
namespace fs=std::filesystem;
void require(int rc, const char* message) {
    if(rc!=0) throw std::runtime_error(message);
}
fs::path directory(const std::string& root) {
    if(root.empty()) throw std::invalid_argument("DTLS automatic mode requires a storage directory");
    fs::path path(root);
    if(fs::is_symlink(path)) throw std::runtime_error("DTLS directory must not be a symlink");
    fs::create_directories(path);
#ifndef _WIN32
    fs::permissions(path,fs::perms::owner_all,fs::perm_options::replace);
#endif
    return path;
}
void save_file(const fs::path& path, const char* contents, std::size_t size) {
    if(fs::is_symlink(path)) throw std::runtime_error("DTLS credential is a symlink");
    // Use restrictive permissions before writing secrets on POSIX.
#ifndef _WIN32
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);
    if(fd<0) throw std::runtime_error("Cannot open DTLS credential file");
    std::size_t written=0;
    while(written<size) {
        const auto n=::write(fd,contents+written,size-written);
        if(n<=0) {::close(fd);throw std::runtime_error("Cannot write DTLS credential file");}
        written+=static_cast<std::size_t>(n);
    }
    if(::close(fd)!=0)throw std::runtime_error("Cannot close DTLS credential file");
#else
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out || !out.write(contents,static_cast<std::streamsize>(size)) || !out.flush())
        throw std::runtime_error("Cannot save DTLS credential file");
#endif
}
std::string hex(const unsigned char* bytes,std::size_t size) {
    static constexpr char symbols[]="0123456789abcdef";
    std::string out;
    out.reserve(size*2);
    for(std::size_t i=0;i<size;++i) {
        out.push_back(symbols[bytes[i]>>4]);
        out.push_back(symbols[bytes[i]&15]);
    }
    return out;
}
std::array<unsigned char,32> hash(const unsigned char* bytes,std::size_t size) {
    std::array<unsigned char,32> digest{};
    require(mbedtls_sha256(bytes,size,digest.data(),0),"DTLS SHA-256 failed");
    return digest;
}
std::string utc(time_t when) {
    std::tm utc{};
#ifdef _WIN32
    if(gmtime_s(&utc,&when)!=0)throw std::runtime_error("DTLS UTC time failed");
#else
    if(!gmtime_r(&when,&utc))throw std::runtime_error("DTLS UTC time failed");
#endif
    char output[16]{};
    if(!std::strftime(output,sizeof(output),"%Y%m%d%H%M%S",&utc))
        throw std::runtime_error("DTLS certificate date failed");
    return output;
}
} // namespace

void local_server(const std::string& root,mbedtls_ctr_drbg_context& random,
                  mbedtls_x509_crt& certificate,mbedtls_pk_context& key) {
    const auto home=directory(root);
    const auto path=home/"server.key";
    if(fs::exists(path)) {
        if(!fs::is_regular_file(path) || fs::is_symlink(path))
            throw std::runtime_error("Unsafe DTLS server key path");
        require(mbedtls_pk_parse_keyfile(&key,path.string().c_str(),nullptr,
                                         mbedtls_ctr_drbg_random,&random),
                "Cannot read existing DTLS server identity; refusing to rotate key");
        if(mbedtls_pk_get_type(&key)!=MBEDTLS_PK_ECKEY)
            throw std::runtime_error("Unsupported automatic DTLS server key");
    } else {
        require(mbedtls_pk_setup(&key,mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)),
                "DTLS EC key initialization failed");
        require(mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1,mbedtls_pk_ec(key),
                                    mbedtls_ctr_drbg_random,&random),
                "DTLS EC server identity generation failed");
        std::array<unsigned char,4096> pem{};
        require(mbedtls_pk_write_key_pem(&key,pem.data(),pem.size()),
                "DTLS private key encoding failed");
        save_file(path,reinterpret_cast<const char*>(pem.data()),
                  std::strlen(reinterpret_cast<const char*>(pem.data())));
    }

    // The certificate rotates, but the private key and therefore its pin do not.
    mbedtls_x509write_cert builder;
    mbedtls_x509write_crt_init(&builder);
    try {
        mbedtls_x509write_crt_set_version(&builder,MBEDTLS_X509_CRT_VERSION_3);
        mbedtls_x509write_crt_set_md_alg(&builder,MBEDTLS_MD_SHA256);
        mbedtls_x509write_crt_set_subject_key(&builder,&key);
        mbedtls_x509write_crt_set_issuer_key(&builder,&key);
        require(mbedtls_x509write_crt_set_subject_name(
            &builder,"CN=VulkanNetwork automatic server"),
            "DTLS subject certificate name failed");
        require(mbedtls_x509write_crt_set_issuer_name(
            &builder,"CN=VulkanNetwork automatic server"),
            "DTLS issuer certificate name failed");
        std::array<unsigned char,16> serial{};
        require(mbedtls_ctr_drbg_random(&random,serial.data(),serial.size()),
                "DTLS certificate serial generation failed");
        serial[0]&=0x7f;
        require(mbedtls_x509write_crt_set_serial_raw(&builder,serial.data(),serial.size()),
                "DTLS certificate serial failed");
        const auto now=std::time(nullptr);
        if(now==-1)throw std::runtime_error("DTLS clock unavailable");
        const auto start=utc(now-86400);
        const auto end=utc(now+365*86400);
        require(mbedtls_x509write_crt_set_validity(&builder,start.c_str(),end.c_str()),
                "DTLS certificate validity failed");
        require(mbedtls_x509write_crt_set_basic_constraints(&builder,0,-1),
                "DTLS basic constraints failed");
        require(mbedtls_x509write_crt_set_key_usage(
            &builder,MBEDTLS_X509_KU_DIGITAL_SIGNATURE),
            "DTLS key usage failed");
        std::array<unsigned char,8192> pem{};
        require(mbedtls_x509write_crt_pem(&builder,pem.data(),pem.size(),
                                         mbedtls_ctr_drbg_random,&random),
                "DTLS certificate generation failed");
        require(mbedtls_x509_crt_parse(&certificate,pem.data(),
                                      std::strlen(reinterpret_cast<const char*>(pem.data()))+1),
                "DTLS certificate parsing failed");
    } catch(...) {
        mbedtls_x509write_crt_free(&builder);
        throw;
    }
    mbedtls_x509write_crt_free(&builder);
}

bool verify_or_pin(const std::string& root,const std::string& peer_id,
                   const mbedtls_x509_crt& certificate) {
    if(peer_id.empty()) return false;
    // Pin the SubjectPublicKeyInfo, not a renewed certificate's signature.
    std::array<unsigned char,2048> der{};
    const int n=mbedtls_pk_write_pubkey_der(
        &certificate.pk,der.data(),der.size());
    if(n<=0 || static_cast<std::size_t>(n)>der.size()) return false;
    const auto pub=hash(der.data()+der.size()-n,static_cast<std::size_t>(n));
    const auto expected=hex(pub.data(),pub.size());
    const auto name=hash(
        reinterpret_cast<const unsigned char*>(peer_id.data()),peer_id.size());
    const auto folder=directory(root)/"pins";
    directory(folder.string());
    const auto path=folder/(hex(name.data(),name.size())+".pin");
    if(fs::is_symlink(path))return false;
    if(fs::exists(path)) {
        std::ifstream in(path,std::ios::binary);
        std::string known;
        if(!in || !std::getline(in,known))return false;
        return known==expected;
    }
    // First use cannot authenticate the endpoint independently. On later
    // connections a changed server key is rejected.
    const auto value=expected+"\n";
    save_file(path,value.data(),value.size());
    std::cerr<<"DTLS: first-use server key pinned for "<<peer_id
             <<" (SHA256 SPKI "<<expected<<")\n";
    return true;
}
} // namespace net::secure::identity
