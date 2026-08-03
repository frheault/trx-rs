use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;

use super::archive_edit::{self, ArchiveOp};
use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::io::filename::TrxFilename;
use crate::mmap_backing::vec_to_bytes;
use crate::trx_file::{DataArray, DataPerGroup, TrxFile};

/// On-disk width for the `offsets.*` array. The TRX spec accepts both
/// `uint32` and `uint64`; we auto-pick at write time via [`pick_for`] so
/// every-day tractograms (≤ 4 G vertices) stay compact and only genuinely
/// huge files pay the doubled-width cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OffsetsDtype {
    U32,
    // Reserved for the day the in-memory offset representation widens past
    // u32. The reader already accepts `offsets.uint64`; the writer will
    // start emitting it once `pick_for` can return this variant.
    #[allow(dead_code)]
    U64,
}

impl OffsetsDtype {
    /// Pick the narrowest dtype that fits every offset in the slice.
    pub(crate) fn pick_for(offsets: &[u32]) -> Self {
        // The in-memory representation is `&[u32]`, so by definition every
        // value fits in `u32`. We still keep this function as the canonical
        // place to widen the rule if the in-memory type ever changes.
        let _ = offsets;
        OffsetsDtype::U32
    }

    /// Filename suffix written for this dtype (e.g. `"uint64"`).
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            OffsetsDtype::U32 => "uint32",
            OffsetsDtype::U64 => "uint64",
        }
    }

    /// Write an in-memory `u32` offset slice directly to a stream at this width.
    pub(crate) fn write_to_stream<W: std::io::Write>(self, offsets: &[u32], writer: &mut W) -> crate::error::Result<()> {
        match self {
            OffsetsDtype::U32 => {
                writer.write_all(bytemuck::cast_slice(offsets))?;
            }
            OffsetsDtype::U64 => {
                let chunk_size = 8192;
                let mut buf = Vec::with_capacity(chunk_size);
                for chunk in offsets.chunks(chunk_size) {
                    buf.clear();
                    buf.extend(chunk.iter().map(|&o| o as u64));
                    writer.write_all(bytemuck::cast_slice(&buf))?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct TrxArchiveIndex {
    dps: HashMap<String, String>,
    dpv: HashMap<String, String>,
    groups: HashMap<String, String>,
    dpg: HashMap<String, HashMap<String, String>>,
}

/// Load a TRX file from a `.trx` zip archive.
pub fn load_from_zip<P: TrxScalar>(path: &Path) -> Result<TrxFile<P>> {
    crate::io::directory::load_from_zip_impl(path)
}

/// Save a `TrxFile<P>` to a `.trx` zip archive.
///
/// All entries are written uncompressed (Stored). DEFLATE rarely pays off on
/// the float-dominated payload of a TRX file: compression ratios are typically
/// <15% and write time grows substantially. Callers who want to compress the
/// `groups/` entries (uint32 streamline-index lists, which do tend to have
/// runs) can use [`save_to_zip_with`].
pub fn save_to_zip<P: TrxScalar>(trx: &TrxFile<P>, path: &Path) -> Result<()> {
    save_to_zip_with(trx, path, zip::CompressionMethod::Stored)
}

/// Save a `TrxFile<P>` to a `.trx` zip archive, applying `groups_compression`
/// to `groups/` entries only. All other entries (header, positions, offsets,
/// dps, dpv, dpg) are always Stored.
pub fn save_to_zip_with<P: TrxScalar>(
    trx: &TrxFile<P>,
    path: &Path,
    groups_compression: zip::CompressionMethod,
) -> Result<()> {
    let offsets_dtype = OffsetsDtype::pick_for(trx.offsets());
    let file = fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let stored = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true);
    let groups_opts = SimpleFileOptions::default()
        .compression_method(groups_compression)
        .large_file(true);

    // Header
    let header_json = trx.header().to_json()?;
    zip.start_file("header.json", stored)?;
    zip.write_all(header_json.as_bytes())?;

    // Positions
    let pos_filename = format!("positions.3.{}", P::DTYPE.name());
    zip.start_file(&pos_filename, stored)?;
    zip.write_all(trx.positions_bytes())?;

    // Offsets — written at `offsets_dtype`'s width.
    let offsets_filename = format!("offsets.{}", offsets_dtype.suffix());
    zip.start_file(&offsets_filename, stored)?;
    offsets_dtype.write_to_stream(trx.offsets(), &mut zip)?;

    // DPS / DPV — float-heavy, Stored.
    write_data_map(&mut zip, "dps", trx.dps_arrays(), stored)?;
    write_data_map(&mut zip, "dpv", trx.dpv_arrays(), stored)?;

    // Groups — uint32 membership lists; honor caller's compression choice.
    write_data_map(&mut zip, "groups", trx.group_arrays(), groups_opts)?;

    // DPG — tiny per-group scalars, Stored.
    write_dpg_map(&mut zip, "dpg", trx.dpg_arrays(), stored)?;

    zip.finish()?;
    Ok(())
}

pub fn append_dps_to_zip(
    path: &Path,
    dps: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let header = read_header_from_zip(path)?;
    validate_row_count("DPS", dps, header.nb_streamlines as usize)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, arr) in dps {
        let target = data_entry_path("dps", name, arr);
        plan_data_write(
            &index.dps,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn append_dpv_to_zip(
    path: &Path,
    dpv: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let header = read_header_from_zip(path)?;
    validate_row_count("DPV", dpv, header.nb_vertices as usize)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, arr) in dpv {
        let target = data_entry_path("dpv", name, arr);
        plan_data_write(
            &index.dpv,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn append_groups_to_zip(
    path: &Path,
    groups: &HashMap<String, Vec<u32>>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, members) in groups {
        validate_group_members(name, members, header.nb_streamlines as usize)?;
        let target = format!("groups/{name}.uint32");
        let bytes = vec_to_bytes(members.clone());
        plan_bytes_write(
            index.groups.get(name),
            target,
            bytes,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn append_dpg_to_zip(
    path: &Path,
    dpg: &DataPerGroup,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let index = build_archive_index(path)?;
    let available_groups: BTreeSet<&str> = index.groups.keys().map(String::as_str).collect();
    let mut ops = Vec::new();
    for (group, arrays) in dpg {
        if !available_groups.contains(group.as_str()) {
            return Err(TrxError::Argument(format!(
                "cannot add DPG entries for missing group '{group}'"
            )));
        }
        let existing = index.dpg.get(group);
        for (name, arr) in arrays {
            let target = format!("dpg/{group}/{}", filename_for_array(name, arr));
            let existing_path = existing.and_then(|entries| entries.get(name));
            plan_bytes_write(
                existing_path,
                target,
                arr.as_bytes().to_vec(),
                overwrite,
                compression,
                &mut ops,
            )?;
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_dps_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for name in names {
        if let Some(entry_path) = index.dps.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_dpv_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for name in names {
        if let Some(entry_path) = index.dpv.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_groups_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for name in names {
        if let Some(entry_path) = index.groups.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
        ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{name}"),
        });
    }
    archive_edit::apply_archive_ops(path, ops)
}

pub fn delete_dpg_from_zip(path: &Path, group: &str, names: Option<&[&str]>) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    match names {
        None => ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{group}"),
        }),
        Some([]) => ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{group}"),
        }),
        Some(names) => {
            if let Some(entries) = index.dpg.get(group) {
                for name in names {
                    if let Some(entry_path) = entries.get(*name) {
                        ops.push(ArchiveOp::Delete {
                            path: entry_path.clone(),
                        });
                    }
                }
            }
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

fn read_header_from_zip(path: &Path) -> Result<Header> {
    let bytes = archive_edit::read_archive_entry(path, "header.json")?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn build_archive_index(path: &Path) -> Result<TrxArchiveIndex> {
    let entries = archive_edit::archive_entry_names(path)?;
    let mut index = TrxArchiveIndex::default();

    for entry in entries {
        if let Some(rest) = entry.strip_prefix("dps/") {
            index_entry(&mut index.dps, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("dpv/") {
            index_entry(&mut index.dpv, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("groups/") {
            index_entry(&mut index.groups, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("dpg/") {
            if let Some((group, file_name)) = rest.split_once('/') {
                let parsed = TrxFilename::parse(file_name)?;
                let group_entries = index.dpg.entry(group.to_string()).or_default();
                if group_entries.insert(parsed.name, entry.clone()).is_some() {
                    return Err(TrxError::Format(format!(
                        "duplicate DPG entry path for group '{group}'"
                    )));
                }
            }
        }
    }

    Ok(index)
}

fn index_entry(
    index: &mut HashMap<String, String>,
    full_path: &str,
    file_name: &str,
) -> Result<()> {
    if file_name.ends_with('/') {
        return Ok(());
    }
    let parsed = TrxFilename::parse(file_name)?;
    if index.insert(parsed.name, full_path.to_string()).is_some() {
        return Err(TrxError::Format(format!(
            "duplicate archive entry for '{full_path}'"
        )));
    }
    Ok(())
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

fn data_entry_path(prefix: &str, name: &str, arr: &DataArray) -> String {
    format!("{prefix}/{}", filename_for_array(name, arr))
}

fn filename_for_array(name: &str, arr: &DataArray) -> String {
    TrxFilename {
        name: name.to_string(),
        ncols: arr.ncols(),
        dtype: arr.dtype(),
    }
    .to_filename()
}

fn plan_data_write(
    existing: &HashMap<String, String>,
    logical_name: &str,
    target_path: String,
    arr: &DataArray,
    overwrite: bool,
    compression: zip::CompressionMethod,
    ops: &mut Vec<ArchiveOp>,
) -> Result<()> {
    plan_bytes_write(
        existing.get(logical_name),
        target_path,
        arr.as_bytes().to_vec(),
        overwrite,
        compression,
        ops,
    )
}

fn plan_bytes_write(
    existing_path: Option<&String>,
    target_path: String,
    bytes: Vec<u8>,
    overwrite: bool,
    compression: zip::CompressionMethod,
    ops: &mut Vec<ArchiveOp>,
) -> Result<()> {
    match existing_path {
        None => ops.push(ArchiveOp::Add {
            path: target_path,
            bytes,
            compression,
        }),
        Some(_) if !overwrite => {}
        Some(existing) if existing == &target_path => ops.push(ArchiveOp::Replace {
            path: target_path,
            bytes,
            compression,
        }),
        Some(existing) => {
            ops.push(ArchiveOp::Delete {
                path: existing.clone(),
            });
            ops.push(ArchiveOp::Add {
                path: target_path,
                bytes,
                compression,
            });
        }
    }
    Ok(())
}

fn write_data_map<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    prefix: &str,
    arrays: &HashMap<String, DataArray>,
    options: SimpleFileOptions,
) -> Result<()> {
    for (name, arr) in arrays {
        let filename = filename_for_array(name, arr);
        let entry_name = format!("{prefix}/{filename}");
        zip.start_file(&entry_name, options)?;
        zip.write_all(arr.as_bytes())?;
    }
    Ok(())
}

fn write_dpg_map<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    prefix: &str,
    groups: &DataPerGroup,
    options: SimpleFileOptions,
) -> Result<()> {
    for (group, arrays) in groups {
        for (name, arr) in arrays {
            let filename = filename_for_array(name, arr);
            let entry_name = format!("{prefix}/{group}/{filename}");
            zip.start_file(&entry_name, options)?;
            zip.write_all(arr.as_bytes())?;
        }
    }
    Ok(())
}
