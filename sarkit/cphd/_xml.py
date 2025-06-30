"""
Functions for interacting with CPHD XML
"""

import copy
import re
from collections.abc import Sequence

import lxml.etree

import sarkit._xmlhelp as skxml
import sarkit.cphd._io as cphd_io


# The following transcoders happen to share common implementation across several standards
@skxml.inheritdocstring
class TxtType(skxml.TxtType):
    pass


@skxml.inheritdocstring
class EnuType(skxml.EnuType):
    pass


@skxml.inheritdocstring
class BoolType(skxml.BoolType):
    pass


@skxml.inheritdocstring
class XdtType(skxml.XdtType):
    pass


@skxml.inheritdocstring
class IntType(skxml.IntType):
    pass


@skxml.inheritdocstring
class DblType(skxml.DblType):
    pass


@skxml.inheritdocstring
class HexType(skxml.HexType):
    pass


@skxml.inheritdocstring
class LineSampType(skxml.LineSampType):
    pass


@skxml.inheritdocstring
class XyType(skxml.XyType):
    pass


@skxml.inheritdocstring
class XyzType(skxml.XyzType):
    pass


@skxml.inheritdocstring
class LatLonType(skxml.LatLonType):
    pass


@skxml.inheritdocstring
class LatLonHaeType(skxml.LatLonHaeType):
    pass


@skxml.inheritdocstring
class PolyType(skxml.PolyType):
    pass


@skxml.inheritdocstring
class Poly2dType(skxml.Poly2dType):
    pass


@skxml.inheritdocstring
class XyzPolyType(skxml.XyzPolyType):
    pass


@skxml.inheritdocstring
class ParameterType(skxml.ParameterType):
    pass


class ImageAreaCornerPointsType(skxml.NdArrayType):
    """
    Transcoder for CPHD-like SceneCoordinates/ImageAreaCornerPoints XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__("IACP", skxml.LatLonType(), include_size_attr=False)

    def set_elem(
        self, elem: lxml.etree.Element, val: Sequence[Sequence[float]]
    ) -> None:
        """Set the IACP children of ``elem`` using the ordered vertices from ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : (4, 2) array_like
            Array of [latitude (deg), longitude (deg)] image corners.

        """
        if len(val) != 4:
            raise ValueError(f"Must have 4 corner points (given {len(val)})")
        super().set_elem(elem, val)


class PvpType(skxml.SequenceType):
    """
    Transcoder for per-vector parameter (PVP) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__(
            {
                "Offset": skxml.IntType(),
                "Size": skxml.IntType(),
                "Format": skxml.TxtType(),
            }
        )

    def parse_elem(self, elem: lxml.etree.Element) -> dict:
        """Returns a dict containing the sequence of subelements encoded in ``elem``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to parse

        Returns
        -------
        elem_dict : dict
            Subelement values by name:

            * "Name" : `str` (`AddedPvpType` only)
            * "Offset" : `int`
            * "Size" : `int`
            * "dtype" : `numpy.dtype`
        """
        elem_dict = super().parse_subelements(elem)
        elem_dict["dtype"] = cphd_io.binary_format_string_to_dtype(elem_dict["Format"])
        del elem_dict["Format"]
        return elem_dict

    def set_elem(self, elem: lxml.etree.Element, val: dict) -> None:
        """Sets ``elem`` node using the sequence of subelements in the dict ``val``.

        Parameters
        ----------
        elem : lxml.etree.Element
            XML element to set
        val : dict
            Subelement values by name:

            * "Name" : `str` (`AddedPvpType` only)
            * "Offset" : `int`
            * "Size" : `int`
            * "dtype" : `numpy.dtype`
        """
        local_val = copy.deepcopy(val)
        local_val["Format"] = cphd_io.dtype_to_binary_format_string(local_val["dtype"])
        del local_val["dtype"]
        super().set_subelements(elem, local_val)


class AddedPvpType(PvpType):
    """
    Transcoder for added per-vector parameter (APVP) XML parameter types.

    """

    def __init__(self) -> None:
        super().__init__()
        self.subelements = {"Name": skxml.TxtType(), **self.subelements}


TRANSCODERS: dict[str, skxml.Type] = {
    "CollectionID/CollectorName": TxtType(),
    "CollectionID/IlluminatorName": TxtType(),
    "CollectionID/CoreName": TxtType(),
    "CollectionID/CollectType": TxtType(),
    "CollectionID/RadarMode/ModeType": TxtType(),
    "CollectionID/RadarMode/ModeID": TxtType(),
    "CollectionID/Classification": TxtType(),
    "CollectionID/ReleaseInfo": TxtType(),
    "CollectionID/CountryCode": TxtType(),
    "CollectionID/Parameter": ParameterType(),
}
TRANSCODERS |= {
    "Global/DomainType": TxtType(),
    "Global/SGN": IntType(),
    "Global/Timeline/CollectionStart": XdtType(),
    "Global/Timeline/RcvCollectionStart": XdtType(),
    "Global/Timeline/TxTime1": DblType(),
    "Global/Timeline/TxTime2": DblType(),
    "Global/FxBand/FxMin": DblType(),
    "Global/FxBand/FxMax": DblType(),
    "Global/TOASwath/TOAMin": DblType(),
    "Global/TOASwath/TOAMax": DblType(),
    "Global/TropoParameters/N0": DblType(),
    "Global/TropoParameters/RefHeight": TxtType(),
    "Global/IonoParameters/TECV": DblType(),
    "Global/IonoParameters/F2Height": DblType(),
}
TRANSCODERS |= {
    "SceneCoordinates/EarthModel": TxtType(),
    "SceneCoordinates/IARP/ECF": XyzType(),
    "SceneCoordinates/IARP/LLH": LatLonHaeType(),
    "SceneCoordinates/ReferenceSurface/Planar/uIAX": XyzType(),
    "SceneCoordinates/ReferenceSurface/Planar/uIAY": XyzType(),
    "SceneCoordinates/ReferenceSurface/HAE/uIAXLL": LatLonType(),
    "SceneCoordinates/ReferenceSurface/HAE/uIAYLL": LatLonType(),
    "SceneCoordinates/ImageArea/X1Y1": XyType(),
    "SceneCoordinates/ImageArea/X2Y2": XyType(),
    "SceneCoordinates/ImageArea/Polygon": skxml.NdArrayType("Vertex", XyType()),
    "SceneCoordinates/ImageAreaCornerPoints": ImageAreaCornerPointsType(),
    "SceneCoordinates/ExtendedArea/X1Y1": XyType(),
    "SceneCoordinates/ExtendedArea/X2Y2": XyType(),
    "SceneCoordinates/ExtendedArea/Polygon": skxml.NdArrayType("Vertex", XyType()),
    "SceneCoordinates/ImageGrid/Identifier": TxtType(),
    "SceneCoordinates/ImageGrid/IARPLocation": LineSampType(),
    "SceneCoordinates/ImageGrid/IAXExtent/LineSpacing": DblType(),
    "SceneCoordinates/ImageGrid/IAXExtent/FirstLine": IntType(),
    "SceneCoordinates/ImageGrid/IAXExtent/NumLines": IntType(),
    "SceneCoordinates/ImageGrid/IAYExtent/SampleSpacing": DblType(),
    "SceneCoordinates/ImageGrid/IAYExtent/FirstSample": IntType(),
    "SceneCoordinates/ImageGrid/IAYExtent/NumSamples": IntType(),
    "SceneCoordinates/ImageGrid/SegmentList/NumSegments": IntType(),
    "SceneCoordinates/ImageGrid/SegmentList/Segment/Identifier": TxtType(),
    "SceneCoordinates/ImageGrid/SegmentList/Segment/StartLine": IntType(),
    "SceneCoordinates/ImageGrid/SegmentList/Segment/StartSample": IntType(),
    "SceneCoordinates/ImageGrid/SegmentList/Segment/EndLine": IntType(),
    "SceneCoordinates/ImageGrid/SegmentList/Segment/EndSample": IntType(),
    "SceneCoordinates/ImageGrid/SegmentList/Segment/SegmentPolygon": skxml.NdArrayType(
        "SV", LineSampType()
    ),
}
TRANSCODERS |= {
    "Data/SignalArrayFormat": TxtType(),
    "Data/NumBytesPVP": IntType(),
    "Data/NumCPHDChannels": IntType(),
    "Data/SignalCompressionID": TxtType(),
    "Data/Channel/Identifier": TxtType(),
    "Data/Channel/NumVectors": IntType(),
    "Data/Channel/NumSamples": IntType(),
    "Data/Channel/SignalArrayByteOffset": IntType(),
    "Data/Channel/PVPArrayByteOffset": IntType(),
    "Data/Channel/CompressedSignalSize": IntType(),
    "Data/NumSupportArrays": IntType(),
    "Data/SupportArray/Identifier": TxtType(),
    "Data/SupportArray/NumRows": IntType(),
    "Data/SupportArray/NumCols": IntType(),
    "Data/SupportArray/BytesPerElement": IntType(),
    "Data/SupportArray/ArrayByteOffset": IntType(),
}
TRANSCODERS |= {
    "Channel/RefChId": TxtType(),
    "Channel/FXFixedCPHD": BoolType(),
    "Channel/TOAFixedCPHD": BoolType(),
    "Channel/SRPFixedCPHD": BoolType(),
    "Channel/Parameters/Identifier": TxtType(),
    "Channel/Parameters/RefVectorIndex": IntType(),
    "Channel/Parameters/FXFixed": BoolType(),
    "Channel/Parameters/TOAFixed": BoolType(),
    "Channel/Parameters/SRPFixed": BoolType(),
    "Channel/Parameters/SignalNormal": BoolType(),
    "Channel/Parameters/Polarization/TxPol": TxtType(),
    "Channel/Parameters/Polarization/RcvPol": TxtType(),
    "Channel/Parameters/Polarization/TxPolRef/AmpH": DblType(),
    "Channel/Parameters/Polarization/TxPolRef/AmpV": DblType(),
    "Channel/Parameters/Polarization/TxPolRef/PhaseV": DblType(),
    "Channel/Parameters/Polarization/RcvPolRef/AmpH": DblType(),
    "Channel/Parameters/Polarization/RcvPolRef/AmpV": DblType(),
    "Channel/Parameters/Polarization/RcvPolRef/PhaseV": DblType(),
    "Channel/Parameters/FxC": DblType(),
    "Channel/Parameters/FxBW": DblType(),
    "Channel/Parameters/FxBWNoise": DblType(),
    "Channel/Parameters/TOASaved": DblType(),
    "Channel/Parameters/TOAExtended/TOAExtSaved": DblType(),
    "Channel/Parameters/TOAExtended/LFMEclipse/FxEarlyLow": DblType(),
    "Channel/Parameters/TOAExtended/LFMEclipse/FxEarlyHigh": DblType(),
    "Channel/Parameters/TOAExtended/LFMEclipse/FxLateLow": DblType(),
    "Channel/Parameters/TOAExtended/LFMEclipse/FxLateHigh": DblType(),
    "Channel/Parameters/DwellTimes/CODId": TxtType(),
    "Channel/Parameters/DwellTimes/DwellId": TxtType(),
    "Channel/Parameters/DwellTimes/DTAId": TxtType(),
    "Channel/Parameters/DwellTimes/UseDTA": BoolType(),
    "Channel/Parameters/ImageArea/X1Y1": XyType(),
    "Channel/Parameters/ImageArea/X2Y2": XyType(),
    "Channel/Parameters/ImageArea/Polygon": skxml.NdArrayType("Vertex", XyType()),
    "Channel/Parameters/Antenna/TxAPCId": TxtType(),
    "Channel/Parameters/Antenna/TxAPATId": TxtType(),
    "Channel/Parameters/Antenna/RcvAPCId": TxtType(),
    "Channel/Parameters/Antenna/RcvAPATId": TxtType(),
    "Channel/Parameters/TxRcv/TxWFId": TxtType(),
    "Channel/Parameters/TxRcv/RcvId": TxtType(),
    "Channel/Parameters/TgtRefLevel/PTRef": DblType(),
    "Channel/Parameters/NoiseLevel/PNRef": DblType(),
    "Channel/Parameters/NoiseLevel/BNRef": DblType(),
    "Channel/Parameters/NoiseLevel/FxNoiseProfile/Point/Fx": DblType(),
    "Channel/Parameters/NoiseLevel/FxNoiseProfile/Point/PN": DblType(),
    "Channel/AddedParameters/Parameter": ParameterType(),
}
TRANSCODERS |= {
    "PVP/TxTime": PvpType(),
    "PVP/TxPos": PvpType(),
    "PVP/TxVel": PvpType(),
    "PVP/RcvTime": PvpType(),
    "PVP/RcvPos": PvpType(),
    "PVP/RcvVel": PvpType(),
    "PVP/SRPPos": PvpType(),
    "PVP/AmpSF": PvpType(),
    "PVP/aFDOP": PvpType(),
    "PVP/aFRR1": PvpType(),
    "PVP/aFRR2": PvpType(),
    "PVP/FX1": PvpType(),
    "PVP/FX2": PvpType(),
    "PVP/FXN1": PvpType(),
    "PVP/FXN2": PvpType(),
    "PVP/TOA1": PvpType(),
    "PVP/TOA2": PvpType(),
    "PVP/TOAE1": PvpType(),
    "PVP/TOAE2": PvpType(),
    "PVP/TDTropoSRP": PvpType(),
    "PVP/TDIonoSRP": PvpType(),
    "PVP/SC0": PvpType(),
    "PVP/SCSS": PvpType(),
    "PVP/SIGNAL": PvpType(),
    "PVP/TxAntenna/TxACX": PvpType(),
    "PVP/TxAntenna/TxACY": PvpType(),
    "PVP/TxAntenna/TxEB": PvpType(),
    "PVP/RcvAntenna/RcvACX": PvpType(),
    "PVP/RcvAntenna/RcvACY": PvpType(),
    "PVP/RcvAntenna/RcvEB": PvpType(),
    "PVP/AddedPVP": AddedPvpType(),
}
TRANSCODERS |= {
    "SupportArray/IAZArray/Identifier": TxtType(),
    "SupportArray/IAZArray/ElementFormat": TxtType(),
    "SupportArray/IAZArray/X0": DblType(),
    "SupportArray/IAZArray/Y0": DblType(),
    "SupportArray/IAZArray/XSS": DblType(),
    "SupportArray/IAZArray/YSS": DblType(),
    "SupportArray/IAZArray/NODATA": HexType(),
    "SupportArray/AntGainPhase/Identifier": TxtType(),
    "SupportArray/AntGainPhase/ElementFormat": TxtType(),
    "SupportArray/AntGainPhase/X0": DblType(),
    "SupportArray/AntGainPhase/Y0": DblType(),
    "SupportArray/AntGainPhase/XSS": DblType(),
    "SupportArray/AntGainPhase/YSS": DblType(),
    "SupportArray/AntGainPhase/NODATA": HexType(),
    "SupportArray/DwellTimeArray/Identifier": TxtType(),
    "SupportArray/DwellTimeArray/ElementFormat": TxtType(),
    "SupportArray/DwellTimeArray/X0": DblType(),
    "SupportArray/DwellTimeArray/Y0": DblType(),
    "SupportArray/DwellTimeArray/XSS": DblType(),
    "SupportArray/DwellTimeArray/YSS": DblType(),
    "SupportArray/DwellTimeArray/NODATA": HexType(),
    "SupportArray/AddedSupportArray/Identifier": TxtType(),
    "SupportArray/AddedSupportArray/ElementFormat": TxtType(),
    "SupportArray/AddedSupportArray/X0": DblType(),
    "SupportArray/AddedSupportArray/Y0": DblType(),
    "SupportArray/AddedSupportArray/XSS": DblType(),
    "SupportArray/AddedSupportArray/YSS": DblType(),
    "SupportArray/AddedSupportArray/NODATA": HexType(),
    "SupportArray/AddedSupportArray/XUnits": TxtType(),
    "SupportArray/AddedSupportArray/YUnits": TxtType(),
    "SupportArray/AddedSupportArray/ZUnits": TxtType(),
    "SupportArray/AddedSupportArray/Parameter": ParameterType(),
}
TRANSCODERS |= {
    "Dwell/NumCODTimes": IntType(),
    "Dwell/CODTime/Identifier": TxtType(),
    "Dwell/CODTime/CODTimePoly": Poly2dType(),
    "Dwell/NumDwellTimes": IntType(),
    "Dwell/DwellTime/Identifier": TxtType(),
    "Dwell/DwellTime/DwellTimePoly": Poly2dType(),
}
TRANSCODERS |= {
    "ReferenceGeometry/SRP/ECF": XyzType(),
    "ReferenceGeometry/SRP/IAC": XyzType(),
    "ReferenceGeometry/ReferenceTime": DblType(),
    "ReferenceGeometry/SRPCODTime": DblType(),
    "ReferenceGeometry/SRPDwellTime": DblType(),
    "ReferenceGeometry/Monostatic/ARPPos": XyzType(),
    "ReferenceGeometry/Monostatic/ARPVel": XyzType(),
    "ReferenceGeometry/Monostatic/SideOfTrack": TxtType(),
    "ReferenceGeometry/Monostatic/SlantRange": DblType(),
    "ReferenceGeometry/Monostatic/GroundRange": DblType(),
    "ReferenceGeometry/Monostatic/DopplerConeAngle": DblType(),
    "ReferenceGeometry/Monostatic/GrazeAngle": DblType(),
    "ReferenceGeometry/Monostatic/IncidenceAngle": DblType(),
    "ReferenceGeometry/Monostatic/AzimuthAngle": DblType(),
    "ReferenceGeometry/Monostatic/TwistAngle": DblType(),
    "ReferenceGeometry/Monostatic/SlopeAngle": DblType(),
    "ReferenceGeometry/Monostatic/LayoverAngle": DblType(),
    "ReferenceGeometry/Bistatic/AzimuthAngle": DblType(),
    "ReferenceGeometry/Bistatic/AzimuthAngleRate": DblType(),
    "ReferenceGeometry/Bistatic/BistaticAngle": DblType(),
    "ReferenceGeometry/Bistatic/BistaticAngleRate": DblType(),
    "ReferenceGeometry/Bistatic/GrazeAngle": DblType(),
    "ReferenceGeometry/Bistatic/TwistAngle": DblType(),
    "ReferenceGeometry/Bistatic/SlopeAngle": DblType(),
    "ReferenceGeometry/Bistatic/LayoverAngle": DblType(),
}
for d in ("Tx", "Rcv"):
    TRANSCODERS |= {
        f"ReferenceGeometry/Bistatic/{d}Platform/Time": DblType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/Pos": XyzType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/Vel": XyzType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/SideOfTrack": TxtType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/SlantRange": DblType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/GroundRange": DblType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/DopplerConeAngle": DblType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/GrazeAngle": DblType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/IncidenceAngle": DblType(),
        f"ReferenceGeometry/Bistatic/{d}Platform/AzimuthAngle": DblType(),
    }
TRANSCODERS |= {
    "Antenna/NumACFs": IntType(),
    "Antenna/NumAPCs": IntType(),
    "Antenna/NumAntPats": IntType(),
    "Antenna/AntCoordFrame/Identifier": TxtType(),
    "Antenna/AntCoordFrame/XAxisPoly": XyzPolyType(),
    "Antenna/AntCoordFrame/YAxisPoly": XyzPolyType(),
    "Antenna/AntCoordFrame/UseACFPVP": BoolType(),
    "Antenna/AntPhaseCenter/Identifier": TxtType(),
    "Antenna/AntPhaseCenter/ACFId": TxtType(),
    "Antenna/AntPhaseCenter/APCXYZ": XyzType(),
    "Antenna/AntPattern/Identifier": TxtType(),
    "Antenna/AntPattern/FreqZero": DblType(),
    "Antenna/AntPattern/GainZero": DblType(),
    "Antenna/AntPattern/EBFreqShift": BoolType(),
    "Antenna/AntPattern/EBFreqShiftSF/DCXSF": DblType(),
    "Antenna/AntPattern/EBFreqShiftSF/DCYSF": DblType(),
    "Antenna/AntPattern/MLFreqDilation": BoolType(),
    "Antenna/AntPattern/MLFreqDilationSF/DCXSF": DblType(),
    "Antenna/AntPattern/MLFreqDilationSF/DCYSF": DblType(),
    "Antenna/AntPattern/GainBSPoly": PolyType(),
    "Antenna/AntPattern/AntPolRef/AmpX": DblType(),
    "Antenna/AntPattern/AntPolRef/AmpY": DblType(),
    "Antenna/AntPattern/AntPolRef/PhaseY": DblType(),
    "Antenna/AntPattern/EB/DCXPoly": PolyType(),
    "Antenna/AntPattern/EB/DCYPoly": PolyType(),
    "Antenna/AntPattern/EB/UseEBPVP": BoolType(),
    "Antenna/AntPattern/Array/GainPoly": Poly2dType(),
    "Antenna/AntPattern/Array/PhasePoly": Poly2dType(),
    "Antenna/AntPattern/Array/AntGPId": TxtType(),
    "Antenna/AntPattern/Element/GainPoly": Poly2dType(),
    "Antenna/AntPattern/Element/PhasePoly": Poly2dType(),
    "Antenna/AntPattern/Element/AntGPId": TxtType(),
    "Antenna/AntPattern/GainPhaseArray/Freq": DblType(),
    "Antenna/AntPattern/GainPhaseArray/ArrayId": TxtType(),
    "Antenna/AntPattern/GainPhaseArray/ElementId": TxtType(),
}
TRANSCODERS |= {
    "TxRcv/NumTxWFs": IntType(),
    "TxRcv/TxWFParameters/Identifier": TxtType(),
    "TxRcv/TxWFParameters/PulseLength": DblType(),
    "TxRcv/TxWFParameters/RFBandwidth": DblType(),
    "TxRcv/TxWFParameters/FreqCenter": DblType(),
    "TxRcv/TxWFParameters/LFMRate": DblType(),
    "TxRcv/TxWFParameters/Polarization": TxtType(),
    "TxRcv/TxWFParameters/Power": DblType(),
    "TxRcv/NumRcvs": IntType(),
    "TxRcv/RcvParameters/Identifier": TxtType(),
    "TxRcv/RcvParameters/WindowLength": DblType(),
    "TxRcv/RcvParameters/SampleRate": DblType(),
    "TxRcv/RcvParameters/IFFilterBW": DblType(),
    "TxRcv/RcvParameters/FreqCenter": DblType(),
    "TxRcv/RcvParameters/LFMRate": DblType(),
    "TxRcv/RcvParameters/Polarization": TxtType(),
    "TxRcv/RcvParameters/PathGain": DblType(),
}


def _decorr_type(xml_path):
    return {f"{xml_path}/{x}": DblType() for x in ("CorrCoefZero", "DecorrRate")}


TRANSCODERS |= {
    "ErrorParameters/Monostatic/PosVelErr/Frame": TxtType(),
    "ErrorParameters/Monostatic/PosVelErr/P1": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/P2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/P3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/V1": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/V2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/V3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P1P2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P1P3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P1V1": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P1V2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P1V3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P2P3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P2V1": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P2V2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P2V3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P3V1": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P3V2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/P3V3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/V1V2": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/V1V3": DblType(),
    "ErrorParameters/Monostatic/PosVelErr/CorrCoefs/V2V3": DblType(),
    **_decorr_type("ErrorParameters/Monostatic/PosVelErr/PositionDecorr"),
    "ErrorParameters/Monostatic/RadarSensor/RangeBias": DblType(),
    "ErrorParameters/Monostatic/RadarSensor/ClockFreqSF": DblType(),
    "ErrorParameters/Monostatic/RadarSensor/CollectionStartTime": DblType(),
    **_decorr_type("ErrorParameters/Monostatic/RadarSensor/RangeBiasDecorr"),
    "ErrorParameters/Monostatic/TropoError/TropoRangeVertical": DblType(),
    "ErrorParameters/Monostatic/TropoError/TropoRangeSlant": DblType(),
    **_decorr_type("ErrorParameters/Monostatic/TropoError/TropoRangeDecorr"),
    "ErrorParameters/Monostatic/IonoError/IonoRangeVertical": DblType(),
    "ErrorParameters/Monostatic/IonoError/IonoRangeRateVertical": DblType(),
    "ErrorParameters/Monostatic/IonoError/IonoRgRgRateCC": DblType(),
    **_decorr_type("ErrorParameters/Monostatic/IonoError/IonoRangeVertDecorr"),
    "ErrorParameters/Monostatic/AddedParameters/Parameter": ParameterType(),
    "ErrorParameters/Bistatic/AddedParameters/Parameter": ParameterType(),
}
for d in ("Tx", "Rcv"):
    TRANSCODERS |= {
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/Frame": TxtType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/P1": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/P2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/P3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/V1": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/V2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/V3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P1P2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P1P3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P1V1": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P1V2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P1V3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P2P3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P2V1": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P2V2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P2V3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P3V1": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P3V2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/P3V3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/V1V2": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/V1V3": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/CorrCoefs/V2V3": DblType(),
        **_decorr_type(
            f"ErrorParameters/Bistatic/{d}Platform/PosVelErr/PositionDecorr"
        ),
        f"ErrorParameters/Bistatic/{d}Platform/RadarSensor/DelayBias": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/RadarSensor/ClockFreqSF": DblType(),
        f"ErrorParameters/Bistatic/{d}Platform/RadarSensor/CollectionStartTime": DblType(),
    }
TRANSCODERS |= {
    "ProductInfo/Profile": TxtType(),
    "ProductInfo/CreationInfo/Application": TxtType(),
    "ProductInfo/CreationInfo/DateTime": XdtType(),
    "ProductInfo/CreationInfo/Site": TxtType(),
    "ProductInfo/CreationInfo/Parameter": ParameterType(),
    "ProductInfo/Parameter": ParameterType(),
}
TRANSCODERS |= {
    "GeoInfo/Desc": ParameterType(),
    "GeoInfo/Point": LatLonType(),
    "GeoInfo/Line": skxml.NdArrayType("Endpoint", LatLonType()),
    "GeoInfo/Polygon": skxml.NdArrayType("Vertex", LatLonType()),
}
TRANSCODERS |= {
    "MatchInfo/NumMatchTypes": IntType(),
    "MatchInfo/MatchType/TypeID": TxtType(),
    "MatchInfo/MatchType/CurrentIndex": IntType(),
    "MatchInfo/MatchType/NumMatchCollections": IntType(),
    "MatchInfo/MatchType/MatchCollection/CoreName": TxtType(),
    "MatchInfo/MatchType/MatchCollection/MatchIndex": IntType(),
    "MatchInfo/MatchType/MatchCollection/Parameter": ParameterType(),
}


# Polynomial subelements
TRANSCODERS.update(
    {
        f"{p}/{coord}": skxml.PolyType()
        for p, v in TRANSCODERS.items()
        if isinstance(v, skxml.XyzPolyType)
        for coord in "XYZ"
    }
)
TRANSCODERS.update(
    {
        f"{p}/Coef": skxml.DblType()
        for p, v in TRANSCODERS.items()
        if isinstance(v, skxml.PolyNdType)
    }
)

# List subelements
TRANSCODERS.update(
    {
        f"{p}/{v.sub_tag}": v.sub_type
        for p, v in TRANSCODERS.items()
        if isinstance(v, skxml.NdArrayType)
    }
)

# Sequence subelements
TRANSCODERS.update(
    {
        f"{p}/{sub_name}": sub_type
        for p, v in TRANSCODERS.items()
        if isinstance(v, skxml.SequenceType)
        for sub_name, sub_type in v.subelements.items()
    }
)


class XmlHelper(skxml.XmlHelper):
    """
    XmlHelper for Compensated Phase History Data (CPHD).

    """

    _transcoders_ = TRANSCODERS

    def _get_simple_path(self, elem):
        return re.sub(r"(GeoInfo/)+", "GeoInfo/", super()._get_simple_path(elem))
