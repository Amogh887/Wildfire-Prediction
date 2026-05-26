"""Coarse lat/lon -> Australian state/territory bounding-box heuristic."""


def region_name(lat: float, lon: float) -> str:
    # ACT (small, check first)
    if -35.95 <= lat <= -35.1 and 148.7 <= lon <= 149.5:
        return "Australian Capital Territory"
    # Tasmania
    if lat <= -39.0 and 143.5 <= lon <= 149.0:
        return "Tasmania"
    # Western Australia
    if lon < 129.0:
        return "Western Australia"
    # Northern Territory vs South Australia (lon 129-138)
    if lon < 138.0:
        return "Northern Territory" if lat > -26.0 else "South Australia"
    # lon 138-141
    if lon < 141.0:
        if lat > -16.0:
            return "Northern Territory"
        return "South Australia" if lat < -26.0 else "South Australia"
    # lon >= 141: QLD / NSW / VIC
    if lat > -29.0:
        return "Queensland"
    if lat > -34.0:
        return "New South Wales"
    return "Victoria"
