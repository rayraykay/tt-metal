// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../cumprod_common.hpp"

#include "dataflow_api.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto compile_time_args{get_compile_time_args()};

    const uint32_t output_tile_byte_count{get_tile_size(static_cast<const DataFormat>(compile_time_args.cb_output))};
    const DataFormat output_data_format{get_dataformat(compile_time_args.cb_output)};
    const InterleavedAddrGenFast<compile_time_args.is_output_dram> addr_gtor{
        .bank_base_address = compile_time_args.dst_addr,
        .page_size = output_tile_byte_count,
        .data_format = output_data_format};

    auto tile_handler{
        [&addr_gtor](
            const uint32_t& batch,
            const uint32_t& channel,
            const uint32_t& ht,
            const uint32_t& wt,
            const CumprodActionArgs& args) -> void { send_tile_from_cb(batch, channel, ht, wt, args, addr_gtor); }};

    for_each_tile_grouped_by_channels(compile_time_args, tile_handler);
}
}  // namespace NAMESPACE
