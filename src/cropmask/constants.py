"""Fixed domain constants for the FAO crop-mask pipeline.

These encode the processing specification and must not be changed without a
corresponding change to the FAO methodology document.
"""

from __future__ import annotations

SQM_PER_ACRE = 4046.8564224

#: The four crops handled by the pipeline, in de-overlap priority order.
#: A crop is erased by every crop that appears before it.
CROP_ORDER = ["Fall Maize", "Sugarcane", "Cotton", "Rice"]

#: Integer code written into the ``predicted`` column of every output layer.
PREDICTED_VALUES = {
    "Rice": 7,
    "Cotton": 2,
    "Fall Maize": 3016,
    "Sugarcane": 1,
}

#: Folder-name spellings that map onto a canonical crop name.
CROP_ALIASES = {
    "Fall Maize": ["fall maize", "fallmaize", "fall_maize", "fall-maize", "maize"],
    "Sugarcane": ["sugarcane", "sugar cane", "sugar_cane", "sugar-cane", "cane"],
    "Cotton": ["cotton"],
    "Rice": ["rice"],
}

#: Sidecar extensions that together make up an ESRI shapefile.
SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx"]

#: Extensions that must be present for a shapefile to be readable at all.
SHAPEFILE_REQUIRED = [".shp", ".shx", ".dbf"]
