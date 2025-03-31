// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "dprint.h"

void kernel_main() {
    DPRINT << "----- Device Print Start ----" << ENDL();
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t dst_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast<true> d = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    // read a tile from start_id
    cb_reserve_back(cb_id_in0, onetile);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    noc_async_read_tile(start_id, s, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, onetile);

    // write 4B to dst buffer using 'noc_inline_dw_write' api
    cb_wait_front(cb_id_in0, onetile);
    // noc_async_write_tile(0, d, get_read_ptr(cb_id_in0));
    tt_l1_ptr int* l1_read_ptr = reinterpret_cast<tt_l1_ptr int*>(get_read_ptr(cb_id_in0));
    uint8_t byte_enable = 0xF;
    auto noc_dst_addr = get_noc_addr(0, d);
    int val = l1_read_ptr[0];
    // Print individual bytes
    DPRINT << "val : " << val << ENDL();
    DPRINT << "noc addr - 0x" << HEX() << noc_dst_addr << ENDL();
    DPRINT << "dram addr encoded in noc_addr - 0x" << HEX() << (noc_dst_addr & 0xFFFFFFFFF) << ENDL();
    noc_inline_dw_write(noc_dst_addr, val, 0xF, noc_index);
    noc_async_write_barrier(noc_index);

    cb_pop_front(cb_id_in0, onetile);
    DPRINT << "----- Device Print End ----" << ENDL();
}
