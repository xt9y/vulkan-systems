#pragma once
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/pk.h>
#include <mbedtls/x509_crt.h>
#include <string>

namespace net::secure::identity {
// Server keys persist across restarts. A new self-signed certificate is
// generated from the same key at each startup (SPKI pins survive renewal).
void local_server(const std::string& directory,
                  mbedtls_ctr_drbg_context& random,
                  mbedtls_x509_crt& certificate,
                  mbedtls_pk_context& key);
// Client-only: trust a server public key on its first connection, and
// reject key changes for the same configured host:port thereafter.
bool verify_or_pin(const std::string& directory,
                   const std::string& peer_id,
                   const mbedtls_x509_crt& certificate);
}
