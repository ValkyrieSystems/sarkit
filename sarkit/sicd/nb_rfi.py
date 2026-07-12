import argparse
import pprint

import sarkit.cphd as skcphd

def main(args=None):
    parser = argparse.ArgumentParser(description="Clean up long-duration low-bandwidth signals in a CPHD")
    parser.add_argument("input_cphd_filename")
    parser.add_argument("output_cphd_filename")
    parser.add_argument("--debug-stem", default="")
    add_rfi_args(parser)
    config = parser.parse_args(args)

    class ProcessChannel(mj.Chain):
        def setup(self, chan_id, median_size, boxcar_size, trim_thresh):
            self["read"] = sar_common_kit.cphd.ReadCphd(config.input_cphd_filename, chan_id)
            signal_buff = self["read"].output("signal")
            pvp_buff = self["read"].output("pvp")

            self["mitigate_rfi"] = RemoveNarrowbandRfi(signal_buff, pvp_buff, median_size, boxcar_size,
                                                       trim_thresh, config.debug_stem)

            self.__operation_outputs__["pvp"] = pvp_buff
            self.__operation_outputs__["signal"] = self["mitigate_rfi"].output("result")

    with skcphd.Reader(config.input_cphd_filename) as r:
        cphd_xmltree = r.metadata.xmltree
        cphd_ew = skcphd.ElementWrapper(cphd_xmltree.getroot())
        ref_chan = cphd_ew.load("./{*}Channel/{*}RefChId")
        assert cphd_ew.load("./{*}Global/{*}DomainType") == "FX"

        per_chan_pvp = {}
        per_chan_signal = {}
        for chan_id in [node.text for node in cphd_ew.cphdroot.xpath("./{*}Data/{*}Channel/{*}Identifier")]:
            name = f"process_{chan_id}"
            chain[name] = ProcessChannel(chan_id, config.median_size, config.boxcar_size,
                                         config.trim_thresh)

            per_chan_pvp[chan_id] = chain[name].output("pvp")
            per_chan_signal[chan_id] = chain[name].output("signal")

    classification = cphd_ew.load("./{*}CollectionID/{*}Classification")
    release_info = cphd_ew.load("./{*}CollectionID/{*}ReleaseInfo")

    cphd_meta = skcphd.Metadata(xmltree=cphd_xmltree)
    with skcphd.Writer(config.output_cphd_filename, cphd_meta) as w:
        for ch_id, pvps in per_chan_pvp.items():
            w.write_pvp(ch_id, pvps)
            w.write_signal(ch_id, per_chan_signal[ch_id]())
        for said_elem in cphd_xmltree.findall(
            "{*}Data/{*}SupportArray/{*}Identifier"
        ):
            w.write_support_array(
                said_elem.text, r.read_support_array(said_elem.text)
            )

        chain["write_cphd"] = sar_common_kit.cphd.WriteCphd(
            config.output_cphd_filename,
            mj.DictionaryOperationOutput(per_chan_pvp),
            mj.DictionaryOperationOutput(per_chan_signal),
            classification,
            release_info,
            ref_chan=ref_chan,
            support_arrays=chain[f"process_{ref_chan}"]["read"].output("support_arrays")
        )
        insights = mj.Orchestrator().run(chain)["insights"]
        pprint.pprint(insights)


def add_rfi_args(parser, prefix=""):
    """Adds RFI arguments to a parser with optional arg prefix."""
    detector_group = parser.add_argument_group("RFI detector arguments")
    detector_group.add_argument(f"--{prefix}median-size", type=int, default=15,
                                help="Size of the 2D median filter used to create the denominator for thresholding")
    detector_group.add_argument(f"--{prefix}boxcar-size", type=int, default=5,
                                help="Size of the 2D boxcar used to create the numerator for thresholding")
    detector_group.add_argument(f"--{prefix}trim-thresh", type=float, default=4.0,
                                help=("Power in a bin, relative to median, above which bins are identified as "
                                      "RFI. Bins with more power than this will be set to zero (default: %(default)g)"))

