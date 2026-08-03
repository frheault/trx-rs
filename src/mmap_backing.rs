use bytemuck::{cast_slice, cast_slice_mut, Pod};
use memmap2::{Mmap, MmapMut};

use crate::error::{Result, TrxError};

/// Convert a `Vec<T: Pod>` into a `Vec<u8>` by copying the raw bytes.
pub fn vec_to_bytes<T: Pod>(v: Vec<T>) -> Vec<u8> {
    cast_slice::<T, u8>(&v).to_vec()
}

/// Owns the backing memory for a TRX data array.
///
/// May be a read-only mmap, a read-write mmap, or an owned heap buffer
/// (used for converted offsets, deep copies, etc.).
pub enum MmapBacking {
    ReadOnly(Mmap),
    ReadOnlySliced { mmap: Mmap, offset: usize, len: usize },
    ReadWrite(MmapMut),
    Owned(Vec<u8>),
}

impl MmapBacking {
    /// Raw bytes view.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            MmapBacking::ReadOnly(m) => m,
            MmapBacking::ReadOnlySliced { mmap, offset, len } => &mmap[*offset..*offset + *len],
            MmapBacking::ReadWrite(m) => m,
            MmapBacking::Owned(v) => v,
        }
    }

    /// Mutable raw bytes view (only for ReadWrite and Owned).
    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        match self {
            MmapBacking::ReadOnly(_) | MmapBacking::ReadOnlySliced { .. } => Err(TrxError::Argument(
                "cannot mutably access read-only mmap".into(),
            )),
            MmapBacking::ReadWrite(m) => Ok(m.as_mut()),
            MmapBacking::Owned(v) => Ok(v.as_mut_slice()),
        }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Whether the backing is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, MmapBacking::ReadOnly(_) | MmapBacking::ReadOnlySliced { .. } | MmapBacking::ReadWrite(_))
    }

    /// Cast the raw bytes to a typed slice.
    ///
    /// Panics if the bytes are not aligned or the length is not a multiple
    /// of `size_of::<T>()`.
    pub fn cast_slice<T: Pod>(&self) -> &[T] {
        cast_slice(self.as_bytes())
    }

    /// Cast the raw bytes to a mutable typed slice.
    pub fn cast_slice_mut<T: Pod>(&mut self) -> Result<&mut [T]> {
        let bytes = self.as_bytes_mut()?;
        Ok(cast_slice_mut(bytes))
    }
}

impl std::fmt::Debug for MmapBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapBacking::ReadOnly(m) => write!(f, "ReadOnly({} bytes)", m.len()),
            MmapBacking::ReadOnlySliced { len, .. } => write!(f, "ReadOnlySliced({} bytes)", len),
            MmapBacking::ReadWrite(m) => write!(f, "ReadWrite({} bytes)", m.len()),
            MmapBacking::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
        }
    }
}
