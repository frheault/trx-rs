use bytemuck::Pod;
use std::collections::HashMap;
use std::path::Path;

use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::typed_view::TypedView2D;

/// Named data array with column count and dtype metadata.
#[derive(Debug)]
pub struct DataArray {
    backing: MmapBacking,
    ncols: usize,
    dtype: DType,
}

impl DataArray {
    pub fn owned_bytes(backing: Vec<u8>, ncols: usize, dtype: DType) -> Self {
        Self {
            backing: MmapBacking::Owned(backing),
            ncols,
            dtype,
        }
    }

    pub(crate) fn from_backing(backing: MmapBacking, ncols: usize, dtype: DType) -> Self {
        Self {
            backing,
            ncols,
            dtype,
        }
    }

    pub fn clone_owned(&self) -> Self {
        Self::owned_bytes(self.backing.as_bytes().to_vec(), self.ncols, self.dtype)
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len_bytes(&self) -> usize {
        self.backing.len()
    }

    pub fn nrows(&self) -> usize {
        let row_bytes = self.ncols * self.dtype.size_of();
        self.len_bytes().checked_div(row_bytes).unwrap_or(0)
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.backing.as_bytes()
    }

    pub(crate) fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        self.backing.as_bytes_mut()
    }

    pub fn cast_slice<T: Pod>(&self) -> &[T] {
        self.backing.cast_slice()
    }

    pub(crate) fn cast_slice_mut<T: Pod>(&mut self) -> Result<&mut [T]> {
        self.backing.cast_slice_mut()
    }

    pub fn typed_view<T: Pod>(&self) -> TypedView2D<'_, T> {
        let data: &[T] = self.cast_slice();
        TypedView2D::new(data, self.ncols)
    }

    /// Convert this array to a `Vec<u32>`, handling int64/uint64 source dtypes.
    pub fn to_u32_vec(&self) -> Vec<u32> {
        match self.dtype {
            DType::UInt32 => self.cast_slice::<u32>().to_vec(),
            DType::Int32 => self.cast_slice::<i32>().iter().map(|&x| x as u32).collect(),
            DType::UInt64 => self.cast_slice::<u64>().iter().map(|&x| x as u32).collect(),
            DType::Int64 => self.cast_slice::<i64>().iter().map(|&x| x as u32).collect(),
            _ => self.cast_slice::<u32>().to_vec(),
        }
    }
}

pub type DataPerGroup = HashMap<String, HashMap<String, DataArray>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataArrayInfo {
    pub ncols: usize,
    pub nrows: usize,
    pub dtype: DType,
}

pub(crate) struct TrxParts {
    pub header: Header,
    pub positions_backing: MmapBacking,
    pub offsets_backing: MmapBacking,
    pub dps: HashMap<String, DataArray>,
    pub dpv: HashMap<String, DataArray>,
    pub groups: HashMap<String, DataArray>,
    pub dpg: DataPerGroup,
    pub tempdir: Option<tempfile::TempDir>,
}

/// Core TRX container, generic over the position scalar type `P`.
///
/// Owns all memory-mapped (or heap-allocated) backings. Typed views borrow
/// from `self` — no explicit lifetime annotations needed.
///
/// # Field ordering
/// `_tempdir` must be declared AFTER mmap fields so it is dropped last,
/// keeping the temp directory alive while mmaps reference files within it.
pub struct TrxFile<P: TrxScalar> {
    header: Header,

    /// Positions backing — `N × 3` elements of type `P`.
    positions_backing: MmapBacking,

    /// Offsets backing — `(nb_streamlines + 1)` u32 values.
    /// TRX offsets are normalized to uint32 internally.
    offsets_backing: MmapBacking,

    /// Data per streamline: name → DataArray with `nb_streamlines` rows.
    dps: HashMap<String, DataArray>,

    /// Data per vertex: name → DataArray with `nb_vertices` rows.
    dpv: HashMap<String, DataArray>,

    /// Groups: name → DataArray of uint32 streamline indices.
    groups: HashMap<String, DataArray>,

    /// Data per group: group name -> field name -> DataArray.
    dpg: DataPerGroup,

    /// Temp directory handle (for zip-extracted files). Kept alive until drop.
    _tempdir: Option<tempfile::TempDir>,

    _phantom: std::marker::PhantomData<P>,
}

impl<P: TrxScalar> TrxFile<P> {
    /// Create an empty TrxFile with the given header.
    pub fn empty(header: Header) -> Self {
        Self {
            header,
            positions_backing: MmapBacking::Owned(Vec::new()),
            offsets_backing: MmapBacking::Owned(Vec::new()),
            dps: HashMap::new(),
            dpv: HashMap::new(),
            groups: HashMap::new(),
            dpg: HashMap::new(),
            _tempdir: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Construct from pre-built components (used by the loader).
    pub(crate) fn from_parts(parts: TrxParts) -> Self {
        Self {
            header: parts.header,
            positions_backing: parts.positions_backing,
            offsets_backing: parts.offsets_backing,
            dps: parts.dps,
            dpv: parts.dpv,
            groups: parts.groups,
            dpg: parts.dpg,
            _tempdir: parts.tempdir,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Return a new `TrxFile` with the header replaced. All data arrays and
    /// positions are shared by reference-counted backing (owned copies).
    pub fn with_updated_header(self, header: Header) -> Self {
        Self { header, ..self }
    }

    // ── Positions ───────────────────────────────────────────────────

    /// Positions as a flat slice of `[P; 3]` arrays.
    pub fn positions(&self) -> &[[P; 3]] {
        self.positions_backing.cast_slice()
    }

    /// Positions as a `TypedView2D` with 3 columns.
    pub fn positions_2d(&self) -> TypedView2D<'_, P> {
        let flat: &[P] = self.positions_backing.cast_slice();
        TypedView2D::new(flat, 3)
    }

    /// Raw position bytes for direct GPU buffer upload.
    pub fn positions_bytes(&self) -> &[u8] {
        self.positions_backing.as_bytes()
    }

    /// Number of vertices (points) across all streamlines.
    pub fn nb_vertices(&self) -> usize {
        self.positions().len()
    }

    // ── Offsets ─────────────────────────────────────────────────────

    /// Offsets as a slice of `u32`. Length is `nb_streamlines + 1`.
    /// The i-th streamline spans `positions[offsets[i]..offsets[i+1]]`.
    pub fn offsets(&self) -> &[u32] {
        if self.header.nb_streamlines == 0 {
            return &[0];
        }
        self.offsets_backing.cast_slice()
    }

    pub fn offsets_vec(&self) -> Vec<u32> {
        self.offsets().to_vec()
    }

    /// Number of streamlines.
    pub fn nb_streamlines(&self) -> usize {
        let offsets = self.offsets();
        if offsets.is_empty() {
            0
        } else {
            offsets.len() - 1
        }
    }

    // ── Streamline access ───────────────────────────────────────────

    /// Get the i-th streamline as a slice of `[P; 3]` points.
    pub fn streamline(&self, i: usize) -> &[[P; 3]] {
        let offsets = self.offsets();
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        &self.positions()[start..end]
    }

    /// Iterate over all streamlines.
    pub fn streamlines(&self) -> StreamlineIter<'_, P> {
        StreamlineIter {
            positions: self.positions(),
            offsets: self.offsets(),
            index: 0,
        }
    }

    /// Length (number of points) of each streamline.
    pub fn streamline_lengths(&self) -> Vec<usize> {
        let offsets = self.offsets();
        offsets.windows(2).map(|w| (w[1] - w[0]) as usize).collect()
    }

    // ── DPS / DPV / Group access ────────────────────────────────────

    /// Get a DPS (data-per-streamline) array cast to type `T`.
    pub fn dps<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self.lookup_dps(name)?;
        Ok(arr.typed_view())
    }

    /// Get a DPV (data-per-vertex) array cast to type `T`.
    pub fn dpv<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self.lookup_dpv(name)?;
        Ok(arr.typed_view())
    }

    /// Get group member indices (always u32).
    pub fn group(&self, name: &str) -> Result<&[u32]> {
        let arr = self.lookup_group(name)?;
        Ok(arr.cast_slice())
    }

    /// Get a DPG (data-per-group) array cast to type `T`.
    pub fn dpg<T: Pod>(&self, group: &str, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self.lookup_dpg(group, name)?;
        Ok(arr.typed_view())
    }

    /// List DPS field names.
    pub fn dps_names(&self) -> Vec<&str> {
        self.dps.keys().map(|s| s.as_str()).collect()
    }

    /// List DPV field names.
    pub fn dpv_names(&self) -> Vec<&str> {
        self.dpv.keys().map(|s| s.as_str()).collect()
    }

    /// List group names.
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }

    /// List DPG group names.
    pub fn dpg_group_names(&self) -> Vec<&str> {
        self.dpg.keys().map(|s| s.as_str()).collect()
    }

    pub fn iter_dps(&self) -> impl Iterator<Item = (&str, DataArrayInfo)> + '_ {
        self.dps
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.info()))
    }

    pub fn iter_dpv(&self) -> impl Iterator<Item = (&str, DataArrayInfo)> + '_ {
        self.dpv
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.info()))
    }

    /// Iterate over groups, yielding `(name, &[u32])` pairs.
    ///
    /// # Safety note
    /// This performs a raw byte reinterpretation via `cast_slice::<u32>()`.
    /// If the on-disk group dtype is **not** `uint32` (e.g. `int64`), the
    /// returned slice will contain mangled values. Prefer
    /// [`group_entries_owned`] which handles dtype conversion correctly.
    pub fn iter_groups(&self) -> impl Iterator<Item = (&str, &[u32])> + '_ {
        self.groups
            .iter()
            .map(|(name, arr)| (name.as_str(), arr.cast_slice::<u32>()))
    }

    pub fn dpg_entries(&self, group: &str) -> Result<Vec<(String, DataArrayInfo)>> {
        let entries = self
            .dpg
            .get(group)
            .ok_or_else(|| TrxError::Argument(format!("no DPG group named '{group}'")))?;
        Ok(entries
            .iter()
            .map(|(name, arr)| (name.clone(), arr.info()))
            .collect())
    }

    pub fn dps_info(&self, name: &str) -> Result<DataArrayInfo> {
        Ok(self.lookup_dps(name)?.info())
    }

    pub fn dps_array(&self, name: &str) -> Result<&DataArray> {
        self.lookup_dps(name)
    }

    pub fn dpv_info(&self, name: &str) -> Result<DataArrayInfo> {
        Ok(self.lookup_dpv(name)?.info())
    }

    pub fn dpv_array(&self, name: &str) -> Result<&DataArray> {
        self.lookup_dpv(name)
    }

    pub fn group_info(&self, name: &str) -> Result<DataArrayInfo> {
        Ok(self.lookup_group(name)?.info())
    }

    pub fn group_array(&self, name: &str) -> Result<&DataArray> {
        self.lookup_group(name)
    }

    pub fn dpg_info(&self, group: &str, name: &str) -> Result<DataArrayInfo> {
        Ok(self.lookup_dpg(group, name)?.info())
    }

    pub fn dpg_array(&self, group: &str, name: &str) -> Result<&DataArray> {
        self.lookup_dpg(group, name)
    }

    pub fn scalar_dps_f32(&self, name: &str) -> Result<Vec<f32>> {
        read_scalar_array_as_f32(self.lookup_dps(name)?, "DPS", name)
    }

    pub fn scalar_dpv_f32(&self, name: &str) -> Result<Vec<f32>> {
        read_scalar_array_as_f32(self.lookup_dpv(name)?, "DPV", name)
    }

    pub fn group_entries_owned(&self) -> Vec<(String, Vec<u32>)> {
        self.groups
            .iter()
            .map(|(name, arr)| (name.clone(), arr.to_u32_vec()))
            .collect()
    }

    // ── Loading (convenience) ───────────────────────────────────────

    /// Load a TRX file from a directory or `.trx` zip archive.
    pub fn load(path: &Path) -> Result<Self> {
        crate::io::load::<P>(path)
    }

    // ── Saving ──────────────────────────────────────────────────────

    /// Save to a directory. The `offsets.*` array is written as `uint32` if
    /// every offset fits in `u32`, otherwise as `uint64`.
    pub fn save_to_directory(&self, path: &Path) -> Result<()> {
        crate::io::directory::save_to_directory(self, path)
    }

    /// Save to a `.trx` zip archive. All entries are Stored (no compression):
    /// DEFLATE rarely pays off on float-heavy tractography data. The
    /// `offsets.*` array width is auto-picked (uint32 if it fits, else uint64).
    pub fn save_to_zip(&self, path: &Path) -> Result<()> {
        crate::io::zip::save_to_zip(self, path)
    }

    /// Save to a `.trx` zip archive, applying DEFLATE only to `groups/` entries.
    /// Everything else is Stored.
    pub fn save_to_zip_deflate_groups(&self, path: &Path) -> Result<()> {
        crate::io::zip::save_to_zip_with(self, path, zip::CompressionMethod::Deflated)
    }

    /// Deprecated alias for [`save_to_zip`] — kept for backward compatibility.
    pub fn save_to_zip_stored(&self, path: &Path) -> Result<()> {
        crate::io::zip::save_to_zip(self, path)
    }

    /// Save — auto-detects format from extension (`.trx` = zip, otherwise directory).
    pub fn save(&self, path: &Path) -> Result<()> {
        if path.extension().and_then(|e| e.to_str()) == Some("trx") {
            self.save_to_zip(path)
        } else {
            self.save_to_directory(path)
        }
    }

    pub fn is_file_backed(&self) -> bool {
        self._tempdir.is_some() && self.positions_backing.is_mapped()
    }

    pub(crate) fn dps_arrays(&self) -> &HashMap<String, DataArray> {
        &self.dps
    }

    pub(crate) fn dpv_arrays(&self) -> &HashMap<String, DataArray> {
        &self.dpv
    }

    pub(crate) fn group_arrays(&self) -> &HashMap<String, DataArray> {
        &self.groups
    }

    pub(crate) fn dpg_arrays(&self) -> &DataPerGroup {
        &self.dpg
    }

    pub(crate) fn dps_arrays_mut(&mut self) -> &mut HashMap<String, DataArray> {
        &mut self.dps
    }

    pub(crate) fn dpv_arrays_mut(&mut self) -> &mut HashMap<String, DataArray> {
        &mut self.dpv
    }

    pub(crate) fn group_arrays_mut(&mut self) -> &mut HashMap<String, DataArray> {
        &mut self.groups
    }

    pub(crate) fn dpg_arrays_mut(&mut self) -> &mut DataPerGroup {
        &mut self.dpg
    }

    pub(crate) fn clone_with_positions_dtype<Q>(&self) -> TrxFile<Q>
    where
        Q: TrxScalar + FromF32,
    {
        let positions: Vec<[Q; 3]> = self
            .positions()
            .iter()
            .map(|point| {
                [
                    Q::from_f32(point[0].to_f32()),
                    Q::from_f32(point[1].to_f32()),
                    Q::from_f32(point[2].to_f32()),
                ]
            })
            .collect();

        TrxFile::from_parts(TrxParts {
            header: self.header.clone(),
            positions_backing: MmapBacking::Owned(vec_to_bytes(positions)),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(self.offsets_vec())),
            dps: clone_data_map(&self.dps),
            dpv: clone_data_map(&self.dpv),
            groups: clone_data_map(&self.groups),
            dpg: clone_dpg_map(&self.dpg),
            tempdir: None,
        })
    }

    fn lookup_dps(&self, name: &str) -> Result<&DataArray> {
        self.dps
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no DPS named '{name}'")))
    }

    fn lookup_dpv(&self, name: &str) -> Result<&DataArray> {
        self.dpv
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no DPV named '{name}'")))
    }

    fn lookup_group(&self, name: &str) -> Result<&DataArray> {
        self.groups
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no group named '{name}'")))
    }

    fn lookup_dpg(&self, group: &str, name: &str) -> Result<&DataArray> {
        let group_map = self
            .dpg
            .get(group)
            .ok_or_else(|| TrxError::Argument(format!("no DPG group named '{group}'")))?;
        group_map
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no DPG named '{name}' in group '{group}'")))
    }
}

impl<P: TrxScalar> std::fmt::Debug for TrxFile<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrxFile")
            .field("dtype", &P::DTYPE)
            .field("nb_streamlines", &self.nb_streamlines())
            .field("nb_vertices", &self.nb_vertices())
            .field("dps", &self.dps_names())
            .field("dpv", &self.dpv_names())
            .field("groups", &self.group_names())
            .field("dpg_group_names", &self.dpg_group_names())
            .finish()
    }
}

/// Iterator over streamlines, yielding `&[[P; 3]]` slices.
pub struct StreamlineIter<'a, P: TrxScalar> {
    positions: &'a [[P; 3]],
    offsets: &'a [u32],
    index: usize,
}

impl<'a, P: TrxScalar> Iterator for StreamlineIter<'a, P> {
    type Item = &'a [[P; 3]];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index + 1 >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[self.index] as usize;
        let end = self.offsets[self.index + 1] as usize;
        self.index += 1;
        Some(&self.positions[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1 - self.index
        };
        (remaining, Some(remaining))
    }
}

impl<'a, P: TrxScalar> ExactSizeIterator for StreamlineIter<'a, P> {}

impl DataArray {
    pub fn info(&self) -> DataArrayInfo {
        DataArrayInfo {
            ncols: self.ncols,
            nrows: self.nrows(),
            dtype: self.dtype,
        }
    }
}

fn read_scalar_array_as_f32(arr: &DataArray, kind: &str, name: &str) -> Result<Vec<f32>> {
    if arr.ncols() != 1 {
        return Err(TrxError::Argument(format!(
            "{kind} '{name}' has {} columns; expected a scalar field",
            arr.ncols()
        )));
    }

    let values = match arr.dtype() {
        DType::Float16 => arr
            .cast_slice::<half::f16>()
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        DType::Float32 => arr.cast_slice::<f32>().to_vec(),
        DType::Float64 => arr
            .cast_slice::<f64>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        DType::Int8 => arr
            .cast_slice::<i8>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        DType::Int16 => arr
            .cast_slice::<i16>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        DType::Int32 => arr
            .cast_slice::<i32>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        DType::UInt8 => arr
            .cast_slice::<u8>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        DType::UInt16 => arr
            .cast_slice::<u16>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        DType::UInt32 => arr
            .cast_slice::<u32>()
            .iter()
            .map(|&value| value as f32)
            .collect(),
        other => {
            return Err(TrxError::DType(format!(
                "{kind} '{name}' uses unsupported scalar dtype {other}"
            )))
        }
    };

    Ok(values)
}

fn clone_data_map(map: &HashMap<String, DataArray>) -> HashMap<String, DataArray> {
    map.iter()
        .map(|(name, arr)| (name.clone(), arr.clone_owned()))
        .collect()
}

fn clone_dpg_map(map: &DataPerGroup) -> DataPerGroup {
    map.iter()
        .map(|(group, entries)| (group.clone(), clone_data_map(entries)))
        .collect()
}

pub(crate) trait FromF32 {
    fn from_f32(value: f32) -> Self;
}

impl FromF32 for half::f16 {
    fn from_f32(value: f32) -> Self {
        half::f16::from_f32(value)
    }
}

impl FromF32 for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl FromF32 for f64 {
    fn from_f32(value: f32) -> Self {
        value as f64
    }
}
