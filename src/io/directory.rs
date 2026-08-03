use bytemuck::cast_slice;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::io::filename::TrxFilename;
use crate::mmap_backing::vec_to_bytes;
use crate::mmap_backing::MmapBacking;
use crate::trx_file::{DataArray, DataPerGroup, TrxFile, TrxParts};

// Re-use `OffsetsDtype` so the directory writer matches the zip writer.
pub(crate) use super::zip::OffsetsDtype;

/// Memory-map a file as read-only.
fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path)?;
    // SAFETY: We trust the file won't be modified externally while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

/// Load data arrays from a subdirectory (e.g. `dps/`, `dpv/`, `groups/`).
fn load_data_dir(dir: &Path) -> Result<HashMap<String, DataArray>> {
    let mut map = HashMap::new();
    if !dir.exists() {
        return Ok(map);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| TrxError::Format(format!("invalid filename: {}", path.display())))?;

        let parsed = TrxFilename::parse(file_name)?;
        let mmap = mmap_file(&path)?;

        map.insert(
            parsed.name.clone(),
            DataArray::from_backing(MmapBacking::ReadOnly(mmap), parsed.ncols, parsed.dtype),
        );
    }

    Ok(map)
}

fn load_dpg_dir(dir: &Path) -> Result<DataPerGroup> {
    let mut out = HashMap::new();
    if !dir.exists() {
        return Ok(out);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let group_name = entry.file_name().to_string_lossy().to_string();
        let data = load_data_dir(&path)?;
        if !data.is_empty() {
            out.insert(group_name, data);
        }
    }

    Ok(out)
}

/// Find a file matching a given name prefix in a directory, regardless of
/// the ncols/dtype suffix.
fn find_file_with_prefix(dir: &Path, prefix: &str) -> Result<std::path::PathBuf> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(prefix) && name_str.chars().nth(prefix.len()) == Some('.') {
            return Ok(entry.path());
        }
    }
    Err(TrxError::FileNotFound(dir.join(prefix)))
}

/// Load a `TrxFile<P>` from an uncompressed directory.
pub fn load_from_directory<P: TrxScalar>(
    dir: &Path,
    tempdir: Option<tempfile::TempDir>,
) -> Result<TrxFile<P>> {
    if !dir.is_dir() {
        return Err(TrxError::FileNotFound(dir.to_path_buf()));
    }

    // Header
    let header = Header::from_file(&dir.join("header.json"))?;

    // Positions
    let pos_path = find_file_with_prefix(dir, "positions")?;
    let pos_fname = pos_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| TrxError::Format("invalid positions filename".into()))?;
    let pos_parsed = TrxFilename::parse(pos_fname)?;

    if pos_parsed.dtype != P::DTYPE {
        return Err(TrxError::DType(format!(
            "expected positions dtype {}, got {}",
            P::DTYPE,
            pos_parsed.dtype
        )));
    }
    if pos_parsed.ncols != 3 {
        return Err(TrxError::Format(format!(
            "positions must have 3 columns, got {}",
            pos_parsed.ncols
        )));
    }

    let positions_backing = MmapBacking::ReadOnly(mmap_file(&pos_path)?);

    // Offsets
    let off_path = find_file_with_prefix(dir, "offsets")?;
    let off_fname = off_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| TrxError::Format("invalid offsets filename".into()))?;
    let off_parsed = TrxFilename::parse(off_fname)?;

    let offsets_mmap = mmap_file(&off_path)?;
    let offsets_backing = convert_offsets_to_u32(
        &offsets_mmap,
        off_parsed.dtype,
        header.nb_streamlines as usize,
        header.nb_vertices as usize,
    )?;

    // DPS, DPV, groups
    let dps = load_data_dir(&dir.join("dps"))?;
    let dpv = load_data_dir(&dir.join("dpv"))?;
    let groups = load_data_dir(&dir.join("groups"))?;
    let dpg = load_dpg_dir(&dir.join("dpg"))?;

    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        dpg,
        tempdir,
    }))
}

/// Convert offset bytes to u32, handling uint64→u32 narrowing and
/// ensuring the sentinel value (nb_vertices) is present.
fn convert_offsets_to_u32(
    mmap: &Mmap,
    dtype: DType,
    nb_streamlines: usize,
    nb_vertices: usize,
) -> Result<MmapBacking> {
    match dtype {
        DType::UInt64 => {
            let values: &[u64] = cast_slice(mmap.as_ref());
            // Check if sentinel is present
            if values.len() == nb_streamlines {
                // Missing sentinel — append nb_vertices
                let mut owned: Vec<u32> = values
                    .iter()
                    .copied()
                    .map(|value| {
                        u32::try_from(value).map_err(|_| {
                            TrxError::Format(format!("offset {value} exceeds uint32 range"))
                        })
                    })
                    .collect::<Result<_>>()?;
                owned.push(nb_vertices as u32);
                let bytes: Vec<u8> = crate::mmap_backing::vec_to_bytes(owned);
                Ok(MmapBacking::Owned(bytes))
            } else if values.len() == nb_streamlines + 1 {
                let owned: Vec<u32> = values
                    .iter()
                    .copied()
                    .map(|value| {
                        u32::try_from(value).map_err(|_| {
                            TrxError::Format(format!("offset {value} exceeds uint32 range"))
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(MmapBacking::Owned(crate::mmap_backing::vec_to_bytes(owned)))
            } else {
                Err(TrxError::Format(format!(
                    "unexpected offset count: {} (expected {} or {})",
                    values.len(),
                    nb_streamlines,
                    nb_streamlines + 1,
                )))
            }
        }
        DType::UInt32 => {
            let values: &[u32] = cast_slice(mmap.as_ref());
            let mut out: Vec<u32> = values.to_vec();
            if out.len() == nb_streamlines {
                out.push(nb_vertices as u32);
            }
            let bytes: Vec<u8> = crate::mmap_backing::vec_to_bytes(out);
            Ok(MmapBacking::Owned(bytes))
        }
        other => Err(TrxError::DType(format!(
            "offsets must be uint32 or uint64, got {other}"
        ))),
    }
}

/// Save a `TrxFile<P>` to an uncompressed directory. The `offsets.*` array
/// width is auto-picked: `uint32` when every offset fits, otherwise `uint64`.
pub fn save_to_directory<P: TrxScalar>(trx: &TrxFile<P>, dir: &Path) -> Result<()> {
    let offsets_dtype = OffsetsDtype::pick_for(trx.offsets());
    fs::create_dir_all(dir)?;

    // Header
    trx.header().write_to(&dir.join("header.json"))?;

    // Positions
    let pos_filename = format!("positions.3.{}", P::DTYPE.name());
    fs::write(dir.join(&pos_filename), trx.positions_bytes())?;

    // Offsets — written at `offsets_dtype`'s width.
    let offsets_filename = format!("offsets.{}", offsets_dtype.suffix());
    let mut file = fs::File::create(dir.join(offsets_filename))?;
    offsets_dtype.write_to_stream(trx.offsets(), &mut file)?;

    // DPS
    save_data_dir(trx.dps_arrays(), &dir.join("dps"))?;

    // DPV
    save_data_dir(trx.dpv_arrays(), &dir.join("dpv"))?;

    // Groups
    save_data_dir(trx.group_arrays(), &dir.join("groups"))?;

    // DPG
    save_dpg_dir(trx.dpg_arrays(), &dir.join("dpg"))?;

    Ok(())
}

pub fn append_dps_to_directory(
    dir: &Path,
    dps: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let header = Header::from_file(&dir.join("header.json"))?;
    validate_row_count("DPS", dps, header.nb_streamlines as usize)?;
    append_arrays_to_directory(&dir.join("dps"), dps, overwrite)
}

pub fn append_dpv_to_directory(
    dir: &Path,
    dpv: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let header = Header::from_file(&dir.join("header.json"))?;
    validate_row_count("DPV", dpv, header.nb_vertices as usize)?;
    append_arrays_to_directory(&dir.join("dpv"), dpv, overwrite)
}

pub fn append_groups_to_directory(
    dir: &Path,
    groups: &HashMap<String, Vec<u32>>,
    overwrite: bool,
) -> Result<()> {
    let header = Header::from_file(&dir.join("header.json"))?;
    let groups_dir = dir.join("groups");
    fs::create_dir_all(&groups_dir)?;
    for (name, members) in groups {
        validate_group_members(name, members, header.nb_streamlines as usize)?;
        let target = groups_dir.join(format!("{name}.uint32"));
        if !overwrite {
            if let Some(existing) = find_named_array_file(&groups_dir, name)? {
                if existing.exists() {
                    continue;
                }
            }
        } else if let Some(existing) = find_named_array_file(&groups_dir, name)? {
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
        }
        fs::write(target, vec_to_bytes(members.clone()))?;
    }
    Ok(())
}

pub fn append_dpg_to_directory(dir: &Path, dpg: &DataPerGroup, overwrite: bool) -> Result<()> {
    let groups_dir = dir.join("groups");
    let dpg_root = dir.join("dpg");
    for (group, entries) in dpg {
        if find_named_array_file(&groups_dir, group)?.is_none() {
            return Err(TrxError::Argument(format!(
                "cannot add DPG entries for missing group '{group}'"
            )));
        }
        let group_dir = dpg_root.join(group);
        fs::create_dir_all(&group_dir)?;
        for (name, arr) in entries {
            let target = group_dir.join(filename_for_array(name, arr));
            if !overwrite {
                if let Some(existing) = find_named_array_file(&group_dir, name)? {
                    if existing.exists() {
                        continue;
                    }
                }
            } else if let Some(existing) = find_named_array_file(&group_dir, name)? {
                if existing != target && existing.exists() {
                    fs::remove_file(existing)?;
                }
            }
            fs::write(target, arr.as_bytes())?;
        }
    }
    Ok(())
}

pub fn delete_dps_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    delete_named_arrays(&dir.join("dps"), names)
}

pub fn delete_dpv_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    delete_named_arrays(&dir.join("dpv"), names)
}

pub fn delete_groups_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    let groups_dir = dir.join("groups");
    for name in names {
        if let Some(path) = find_named_array_file(&groups_dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        let dpg_group = dir.join("dpg").join(name);
        if dpg_group.exists() {
            fs::remove_dir_all(dpg_group)?;
        }
    }
    Ok(())
}

pub fn delete_dpg_from_directory(dir: &Path, group: &str, names: Option<&[&str]>) -> Result<()> {
    let group_dir = dir.join("dpg").join(group);
    match names {
        None | Some([]) => {
            if group_dir.exists() {
                fs::remove_dir_all(group_dir)?;
            }
        }
        Some(names) => {
            for name in names {
                if let Some(path) = find_named_array_file(&group_dir, name)? {
                    if path.exists() {
                        fs::remove_file(path)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn save_data_dir(arrays: &HashMap<String, DataArray>, dir: &Path) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (name, arr) in arrays {
        let filename = filename_for_array(name, arr);
        fs::write(dir.join(&filename), arr.as_bytes())?;
    }
    Ok(())
}

fn save_dpg_dir(arrays: &DataPerGroup, dir: &Path) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (group, entries) in arrays {
        save_data_dir(entries, &dir.join(group))?;
    }
    Ok(())
}

fn append_arrays_to_directory(
    dir: &Path,
    arrays: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    for (name, arr) in arrays {
        let target = dir.join(filename_for_array(name, arr));
        if !overwrite {
            if let Some(existing) = find_named_array_file(dir, name)? {
                if existing.exists() {
                    continue;
                }
            }
        } else if let Some(existing) = find_named_array_file(dir, name)? {
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
        }
        fs::write(target, arr.as_bytes())?;
    }
    Ok(())
}

fn delete_named_arrays(dir: &Path, names: &[&str]) -> Result<()> {
    for name in names {
        if let Some(path) = find_named_array_file(dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn find_named_array_file(dir: &Path, name: &str) -> Result<Option<std::path::PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| TrxError::Format(format!("invalid filename: {}", path.display())))?;
        let parsed = TrxFilename::parse(file_name)?;
        if parsed.name == name {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn validate_row_count(
    kind: &str,
    arrays: &HashMap<String, DataArray>,
    expected_rows: usize,
) -> Result<()> {
    for (name, arr) in arrays {
        if arr.nrows() != expected_rows {
            return Err(TrxError::Format(format!(
                "{kind} '{name}' has {} rows, expected {expected_rows}",
                arr.nrows()
            )));
        }
    }
    Ok(())
}

fn validate_group_members(name: &str, members: &[u32], nb_streamlines: usize) -> Result<()> {
    for &member in members {
        if member as usize >= nb_streamlines {
            return Err(TrxError::Format(format!(
                "group '{name}' contains streamline index {member}, but NB_STREAMLINES is {nb_streamlines}"
            )));
        }
    }
    Ok(())
}

fn filename_for_array(name: &str, arr: &DataArray) -> String {
    TrxFilename {
        name: name.to_string(),
        ncols: arr.ncols(),
        dtype: arr.dtype(),
    }
    .to_filename()
}

pub(crate) fn load_from_zip_impl<P: TrxScalar>(path: &Path) -> Result<TrxFile<P>> {
    let file = fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    
    // Header
    let mut header_file = archive.by_name("header.json")?;
    let mut header_bytes = Vec::new();
    std::io::Read::read_to_end(&mut header_file, &mut header_bytes)?;
    let header: Header = serde_json::from_slice(&header_bytes)?;
    drop(header_file);
    
    // Positions
    let pos_name = (0..archive.len())
        .find_map(|i| {
            let name = archive.name_for_index(i).unwrap().to_string();
            if name.starts_with("positions.") {
                Some(name)
            } else {
                None
            }
        })
        .ok_or_else(|| TrxError::Format("no positions file found in zip archive".into()))?;
        
    let pos_parsed = TrxFilename::parse(&pos_name)?;
    if pos_parsed.dtype != P::DTYPE {
        return Err(TrxError::DType(format!(
            "expected positions dtype {}, got {}",
            P::DTYPE,
            pos_parsed.dtype
        )));
    }
    if pos_parsed.ncols != 3 {
        return Err(TrxError::Format(format!(
            "positions must have 3 columns, got {}",
            pos_parsed.ncols
        )));
    }
    
    let pos_file = archive.by_name(&pos_name)?;
    let pos_data_start = pos_file.data_start();
    let pos_size = pos_file.size();
    let pos_compression = pos_file.compression();
    drop(pos_file);
    
    let file = fs::File::open(path)?;
    
    let is_aligned = |offset: u64, size: usize| -> bool {
        offset % (size as u64) == 0
    };
    
    let page_size = 4096;
    
    let positions_backing = if pos_compression == zip::CompressionMethod::Stored && is_aligned(pos_data_start, std::mem::size_of::<P>()) {
        let align_offset = pos_data_start % page_size;
        let map_offset = pos_data_start - align_offset;
        let map_len = pos_size as usize + align_offset as usize;
        let mmap = unsafe { memmap2::MmapOptions::new().offset(map_offset).len(map_len).map(&file)? };
        MmapBacking::ReadOnlySliced { mmap, offset: align_offset as usize, len: pos_size as usize }
    } else {
        let mut pos_file = archive.by_name(&pos_name)?;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut pos_file, &mut bytes)?;
        MmapBacking::Owned(bytes)
    };
    
    // Offsets
    let off_name = (0..archive.len())
        .find_map(|i| {
            let name = archive.name_for_index(i).unwrap().to_string();
            if name.starts_with("offsets.") {
                Some(name)
            } else {
                None
            }
        })
        .ok_or_else(|| TrxError::Format("no offsets file found in zip archive".into()))?;
        
    let off_parsed = TrxFilename::parse(&off_name)?;
    
    let off_file = archive.by_name(&off_name)?;
    let off_data_start = off_file.data_start();
    let off_size = off_file.size();
    let off_compression = off_file.compression();
    drop(off_file);
    
    let offsets_backing = if off_compression == zip::CompressionMethod::Stored && is_aligned(off_data_start, off_parsed.dtype.size_of()) {
        let align_offset = off_data_start % page_size;
        let map_offset = off_data_start - align_offset;
        let map_len = off_size as usize + align_offset as usize;
        let mmap = unsafe { memmap2::MmapOptions::new().offset(map_offset).len(map_len).map(&file)? };
        
        let slice_len = off_size as usize;
        let slice_offset = align_offset as usize;
        let backed = MmapBacking::ReadOnlySliced { mmap, offset: slice_offset, len: slice_len };
        let mut owned_bytes = Vec::new();
        std::io::Read::read_to_end(&mut std::io::Cursor::new(backed.as_bytes()), &mut owned_bytes)?;
        let mut owned_mmap = memmap2::MmapMut::map_anon(owned_bytes.len())?;
        owned_mmap.copy_from_slice(&owned_bytes);
        
        convert_offsets_to_u32(
            &owned_mmap.make_read_only()?,
            off_parsed.dtype,
            header.nb_streamlines as usize,
            header.nb_vertices as usize,
        )?
    } else {
        let mut off_file = archive.by_name(&off_name)?;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut off_file, &mut bytes)?;
        let mut mmap = memmap2::MmapMut::map_anon(bytes.len())?;
        mmap.copy_from_slice(&bytes);
        convert_offsets_to_u32(
            &mmap.make_read_only()?,
            off_parsed.dtype,
            header.nb_streamlines as usize,
            header.nb_vertices as usize,
        )?
    };
    
    // DPS, DPV, groups, dpg
    let mut dps = HashMap::new();
    let mut dpv = HashMap::new();
    let mut groups = HashMap::new();
    let mut dpg = HashMap::new();
    
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if name.ends_with('/') || name == "header.json" || name.starts_with("positions.") || name.starts_with("offsets.") {
            continue;
        }
        
        let data_start = entry.data_start();
        let size = entry.size();
        let compression = entry.compression();
        let is_stored = compression == zip::CompressionMethod::Stored;
        drop(entry); // Need to drop borrow to access archive
        
        let file = fs::File::open(path)?;
        let load_entry = |archive: &mut zip::ZipArchive<fs::File>, dtype_size: usize| -> Result<MmapBacking> {
            if is_stored && is_aligned(data_start, dtype_size) {
                let align_offset = data_start % page_size;
                let map_offset = data_start - align_offset;
                let map_len = size as usize + align_offset as usize;
                let mmap = unsafe { memmap2::MmapOptions::new().offset(map_offset).len(map_len).map(&file)? };
                Ok(MmapBacking::ReadOnlySliced { mmap, offset: align_offset as usize, len: size as usize })
            } else {
                let mut entry = archive.by_name(&name)?;
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut entry, &mut bytes)?;
                Ok(MmapBacking::Owned(bytes))
            }
        };
        
        if name.starts_with("dps/") {
            let basename = name.strip_prefix("dps/").unwrap();
            let parsed = TrxFilename::parse(basename)?;
            dps.insert(parsed.name, DataArray::from_backing(load_entry(&mut archive, parsed.dtype.size_of())?, parsed.ncols, parsed.dtype));
        } else if name.starts_with("dpv/") {
            let basename = name.strip_prefix("dpv/").unwrap();
            let parsed = TrxFilename::parse(basename)?;
            dpv.insert(parsed.name, DataArray::from_backing(load_entry(&mut archive, parsed.dtype.size_of())?, parsed.ncols, parsed.dtype));
        } else if name.starts_with("groups/") {
            let basename = name.strip_prefix("groups/").unwrap();
            let parsed = TrxFilename::parse(basename)?;
            groups.insert(parsed.name, DataArray::from_backing(load_entry(&mut archive, parsed.dtype.size_of())?, parsed.ncols, parsed.dtype));
        } else if name.starts_with("dpg/") {
            let rest = name.strip_prefix("dpg/").unwrap();
            if let Some((group, basename)) = rest.split_once('/') {
                let parsed = TrxFilename::parse(basename)?;
                dpg.entry(group.to_string())
                    .or_insert_with(HashMap::new)
                    .insert(parsed.name, DataArray::from_backing(load_entry(&mut archive, parsed.dtype.size_of())?, parsed.ncols, parsed.dtype));
            }
        }
    }
    
    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        dpg,
        tempdir: None,
    }))
}
