//! Cross-platform memory-mapped file abstraction.
//!
//! Uses `memmap2` on Unix and Windows for simplicity and safety.

use std::fs::File;
use std::io;
use std::path::Path;

pub struct Mmap {
    inner: memmap2::Mmap,
}

impl Mmap {
    /// Memory-map a file for read-only access.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let inner = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { inner })
    }

    /// Pointer to the start of the mapped region.
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    /// Length of the mapped region in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
