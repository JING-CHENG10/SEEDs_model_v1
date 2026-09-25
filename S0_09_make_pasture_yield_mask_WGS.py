# -*- coding: utf-8 -*-
"""
Use the original vector-rasterization mask logic to match the reference GeoTIFF CRS, resolution, and extent exactly.
- Reference raster: ref_tif_path, e.g. pastures_coi_Area_ha.tif at 1/12 degree.
- Vector boundaries: shp_path with ID1; fall back to a 0/1 mask if absent.
- Output: matching Int32 GTiff, with zero for NoData/ocean/no country.

Dependencies: GDAL (osgeo.gdal/ogr/osr), preferably >=3.4.
"""

import os
import tempfile
from osgeo import gdal, ogr, osr

# Paths to check
# Reference raster defines CRS, extent, and resolution; target cell size is 1/12 degree.
ref_tif_path = r"..\\..\\input\\Land\\Feed_pasture\\pastures_coi_Area_ha.tif"  
# World boundaries with ID1, e.g. Shapefile or GPKG
# Use ID1 as in the original script; otherwise a binary mask with land=1 and ocean=0.
shp_path = r"..\\..\\src\\World_map\\polygon\\World_polygon.shp"  # Replace with your actual path.

# Select GTiff/NetCDF output automatically by file extension.
out_mask_path = r"..\\..\\src\\mask_pastureYield_0083d.nc"


ATTRIBUTE_FIELD = "ID1"       # Prefer ID1 values, as in the original script.
ALL_TOUCHED = True            # Whether rasterization includes all touched cells, expanding boundary coverage
NODATA_VALUE = 0              # Fill ocean/no country with zero, preserving original semantics.
SHAPE_ENCODING = "CP936"      # For Chinese-encoded vector attributes, use CP936/GBK; None preserves the default.
FORCE_UTF8 = False            # Set False to suppress non-UTF-8 warnings.

def _open_vector(vpath):
    vds = gdal.OpenEx(vpath, gdal.OF_VECTOR)
    if vds is None:
        raise RuntimeError(f"无法打开矢量文件：{vpath}")
    lyr = vds.GetLayer(0)
    if lyr is None:
        raise RuntimeError(f"矢量文件无有效图层：{vpath}")
    return vds, lyr

def _srs_from_wkt(wkt):
    srs = osr.SpatialReference()
    if wkt is None or len(wkt.strip()) == 0:
        return None
    srs.ImportFromWkt(wkt)
    return srs

def _srs_equivalent(src_srs, target_srs):
    """Check SRS equivalence, tolerating absent EPSG authority metadata."""
    if src_srs is None or target_srs is None:
        return False
    try:
        if src_srs.IsSame(target_srs):
            return True
    except Exception:
        pass
    try:
        src_clone = src_srs.Clone()
        tgt_clone = target_srs.Clone()
        src_clone.AutoIdentifyEPSG()
        tgt_clone.AutoIdentifyEPSG()
        src_code = src_clone.GetAuthorityCode(None)
        tgt_code = tgt_clone.GetAuthorityCode(None)
        if src_code and tgt_code and src_code == tgt_code:
            return True
    except Exception:
        pass
    try:
        src_proj = (src_srs.ExportToProj4() or "").strip()
        tgt_proj = (target_srs.ExportToProj4() or "").strip()
        if src_proj and tgt_proj and src_proj == tgt_proj:
            return True
    except Exception:
        pass
    name_src = (src_srs.GetAttrValue("GEOGCS") or "").lower()
    name_tgt = (target_srs.GetAttrValue("GEOGCS") or "").lower()
    if "wgs" in name_src and "84" in name_src and "wgs" in name_tgt and "84" in name_tgt:
        return True
    return False

def _vector_reproject_to(vds, target_srs):
    """Reproject vectors to target_srs with GDAL VectorTranslate; return temporary GPKG path and layer."""
    lyr = vds.GetLayer(0)
    src_srs = lyr.GetSpatialRef() if lyr is not None else None
    tmp_dir = tempfile.mkdtemp(prefix="mask_vec_")
    gpkg_path = os.path.join(tmp_dir, "vec.gpkg")

    target_wkt = target_srs.ExportToWkt()
    common_kwargs = dict(geometryType="MULTIPOLYGON")
    if _srs_equivalent(src_srs, target_srs):
        # Override SRS metadata only, avoiding a PROJ transformation.
        opts = gdal.VectorTranslateOptions(srcSRS=target_wkt, dstSRS=target_wkt, reproject=False, **common_kwargs)
    else:
        opts = gdal.VectorTranslateOptions(dstSRS=target_wkt, **common_kwargs)

    vds2 = gdal.VectorTranslate(gpkg_path, vds, options=opts)
    if vds2 is None:
        raise RuntimeError("VectorTranslate 失败（重投影矢量失败）。")
    lyr2 = vds2.GetLayer(0)
    if lyr2 is None:
        raise RuntimeError("重投影后的矢量缺少图层。")
    return gpkg_path, vds2, lyr2, tmp_dir

def _layer_has_field(lyr, field_name):
    defn = lyr.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        if defn.GetFieldDefn(i).GetName().lower() == field_name.lower():
            return True
    return False

def _choose_driver(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return "GTiff", [
            "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6",
            "TILED=YES", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"
        ]
    if ext == ".nc":
        return "NetCDF", [
            "FORMAT=NC4", "COMPRESS=DEFLATE", "ZLEVEL=4"
        ]
    raise RuntimeError(f"暂不支持的输出格式：{path}")

def main():
    if not FORCE_UTF8:
        gdal.SetConfigOption("OGR_FORCE_UTF8", "NO")
    if SHAPE_ENCODING:
        gdal.SetConfigOption("SHAPE_ENCODING", SHAPE_ENCODING)

    # 1) Open the reference raster and read grid parameters.
    rds = gdal.Open(ref_tif_path, gdal.GA_ReadOnly)
    if rds is None:
        raise RuntimeError(f"无法打开参考栅格：{ref_tif_path}")
    gt = rds.GetGeoTransform()
    proj_wkt = rds.GetProjection()
    cols, rows = rds.RasterXSize, rds.RasterYSize
    px_w, px_h = gt[1], abs(gt[5])
    srs_raster = _srs_from_wkt(proj_wkt)
    if srs_raster is None:
        raise RuntimeError("参考栅格缺少投影信息（Projection WKT 为空）。")

    # Report whether cell size is close to 1/12 degree.
    try:
        approx_1_12 = abs(px_w - (1.0/12.0)) < 1e-6 and abs(px_h - (1.0/12.0)) < 1e-6
        if not approx_1_12:
            print(f"[警告] 参考栅格像元大小为 ({px_w}, {px_h})，并非严格1/12°；将仍以参考栅格为准对齐输出。")
    except Exception:
        pass

    # 2) Open vectors and project to the raster SRS.
    vds, lyr = _open_vector(shp_path)
    gpkg_path, vds2, lyr2, tmp_dir = _vector_reproject_to(vds, srs_raster)
    if lyr2 is None:
        raise RuntimeError("矢量图层打开失败。")

    # 3) Create an output raster matching the reference.
    drv_name, creation_opts = _choose_driver(out_mask_path)
    drv = gdal.GetDriverByName(drv_name)
    ods = drv.Create(out_mask_path, cols, rows, 1, gdal.GDT_Int32, options=creation_opts)
    if ods is None:
        raise RuntimeError(f"无法创建输出：{out_mask_path}")
    ods.SetGeoTransform(gt)
    ods.SetProjection(proj_wkt)
    band = ods.GetRasterBand(1)
    band.SetNoDataValue(NODATA_VALUE)
    band.Fill(NODATA_VALUE)

    # 4) Rasterize using ID1, or a binary mask if absent.
    use_attribute = _layer_has_field(lyr2, ATTRIBUTE_FIELD)
    if use_attribute:
        ropts = [f"ATTRIBUTE={ATTRIBUTE_FIELD}"]
        if ALL_TOUCHED:
            ropts.append("ALL_TOUCHED=TRUE")
        err = gdal.RasterizeLayer(ods, [1], lyr2, options=ropts)
        mode = f"ATTRIBUTE={ATTRIBUTE_FIELD}"
    else:
        ropts = [f"BURN_VALUE=1"]
        if ALL_TOUCHED:
            ropts.append("ALL_TOUCHED=TRUE")
        err = gdal.RasterizeLayer(ods, [1], lyr2, burn_values=[1], options=ropts)
        mode = "BINARY(0/1)"

    if err != 0:
        raise RuntimeError(f"RasterizeLayer 失败（mode={mode}）。")

    # 5) Optionally build pyramids for quick viewing.
    if drv_name == "GTiff":
        try:
            ods.BuildOverviews("NEAREST", [2, 4, 8, 16, 32])
        except Exception:
            pass

    # 6) Finalize.
    band = None; ods = None
    vds = None
    vds2 = None
    rds = None
    if tmp_dir:
        try:
            import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    print("完成：", out_mask_path)
    print(f"与参考一致：size=({cols},{rows}), pixel=({px_w},{px_h}), CRS={srs_raster.GetAttrValue('AUTHORITY',1) or 'WKT'}")
    print(f"栅格化方式：{mode}；NoData={NODATA_VALUE}；ALL_TOUCHED={ALL_TOUCHED}")

if __name__ == "__main__":
    # Configure GDAL/PROJ data paths if needed, especially for Windows paths containing Chinese characters.
    # import os
    # os.environ.setdefault('PROJ_LIB', os.environ.get('NZF_PROJ_LIB', ''))
    # os.environ.setdefault('GDAL_DATA', os.environ.get('NZF_GDAL_DATA', ''))
    main()
