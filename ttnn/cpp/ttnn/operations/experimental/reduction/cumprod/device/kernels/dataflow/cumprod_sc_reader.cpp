// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../cumprod_common.hpp"

#include "dataflow_api.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto compile_time_args{get_compile_time_args()};

    const uint32_t input_tile_byte_count{get_tile_size(static_cast<const DataFormat>(compile_time_args.cb_input))};
    const DataFormat input_data_format{get_dataformat(compile_time_args.cb_input)};
    const InterleavedAddrGenFast<compile_time_args.is_input_dram> addr_gtor{
        .bank_base_address = compile_time_args.src_addr,
        .page_size = input_tile_byte_count,
        .data_format = input_data_format};

    auto tile_handler{
        [&addr_gtor](
            const uint32_t& batch,
            const uint32_t& channel,
            const uint32_t& ht,
            const uint32_t& wt,
            const CumprodActionArgs& args) -> void { read_tile_into_cb(batch, channel, ht, wt, args, addr_gtor); }};

    for_each_tile_grouped_by_channels(compile_time_args, tile_handler);
}
}  // namespace NAMESPACE
