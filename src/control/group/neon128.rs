use super::super::{BitMask, Tag};
use core::arch::aarch64 as neon;
use core::mem;
use core::num::NonZeroU64;

pub(crate) type BitMaskWord = u64;
pub(crate) type NonZeroBitMaskWord = NonZeroU64;
pub(crate) const BITMASK_STRIDE: usize = 4;
pub(crate) const BITMASK_MASK: BitMaskWord = !0;
pub(crate) const BITMASK_ITER_MASK: BitMaskWord = 0x8888_8888_8888_8888;

#[inline]
fn cmp_to_word(cmp: neon::uint8x16_t) -> BitMaskWord {
    unsafe {
        let cmp = neon::vreinterpretq_u16_u8(cmp);
        let res = neon::vshrn_n_u16(cmp, 4);
        neon::vget_lane_u64(neon::vreinterpret_u64_u8(res), 0)
    }
}

/// Abstraction over a group of control tags which can be scanned in
/// parallel.
///
/// This implementation uses a 64-bit NEON value.
#[derive(Copy, Clone)]
pub(crate) struct Group(neon::uint8x16_t);

#[allow(clippy::use_self)]
impl Group {
    /// Number of bytes in the group.
    pub(crate) const WIDTH: usize = mem::size_of::<Self>();

    /// Returns a full group of empty tags, suitable for use as the initial
    /// value for an empty hash table.
    ///
    /// This is guaranteed to be aligned to the group size.
    #[inline]
    pub(crate) const fn static_empty() -> &'static [Tag; Group::WIDTH] {
        #[repr(C)]
        struct AlignedTags {
            _align: [Group; 0],
            tags: [Tag; Group::WIDTH],
        }
        const ALIGNED_TAGS: AlignedTags = AlignedTags {
            _align: [],
            tags: [Tag::EMPTY; Group::WIDTH],
        };
        &ALIGNED_TAGS.tags
    }

    /// Loads a group of tags starting at the given address.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)] // unaligned load
    pub(crate) unsafe fn load(ptr: *const Tag) -> Self {
        Group(neon::vld1q_u8(ptr.cast()))
    }

    /// Loads a group of tags starting at the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn load_aligned(ptr: *const Tag) -> Self {
        debug_assert_eq!(ptr.align_offset(mem::align_of::<Self>()), 0);
        Group(neon::vld1q_u8(ptr.cast()))
    }

    /// Stores the group of tags to the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub(crate) unsafe fn store_aligned(self, ptr: *mut Tag) {
        debug_assert_eq!(ptr.align_offset(mem::align_of::<Self>()), 0);
        neon::vst1q_u8(ptr.cast(), self.0);
    }

    /// Returns a `BitMask` indicating all tags in the group which *may*
    /// have the given value.
    #[inline]
    pub(crate) fn match_tag(self, tag: Tag) -> BitMask {
        unsafe {
            let cmp = neon::vceqq_u8(self.0, neon::vdupq_n_u8(tag.0));
            BitMask(cmp_to_word(cmp))
        }
    }

    /// Returns a `BitMask` indicating all tags in the group which are
    /// `EMPTY`.
    #[inline]
    pub(crate) fn match_empty(self) -> BitMask {
        self.match_tag(Tag::EMPTY)
    }

    /// Returns a `BitMask` indicating all tags in the group which are
    /// `EMPTY` or `DELETED`.
    #[inline]
    pub(crate) fn match_empty_or_deleted(self) -> BitMask {
        unsafe {
            let cmp = neon::vcltzq_s8(neon::vreinterpretq_s8_u8(self.0));
            BitMask(cmp_to_word(cmp))
        }
    }

    /// Returns a `BitMask` indicating all tags in the group which are full.
    #[inline]
    pub(crate) fn match_full(self) -> BitMask {
        unsafe {
            let cmp = neon::vcgezq_s8(neon::vreinterpretq_s8_u8(self.0));
            BitMask(cmp_to_word(cmp))
        }
    }

    /// Performs the following transformation on all tags in the group:
    /// - `EMPTY => EMPTY`
    /// - `DELETED => EMPTY`
    /// - `FULL => DELETED`
    #[inline]
    pub(crate) fn convert_special_to_empty_and_full_to_deleted(self) -> Self {
        // Map high_bit = 1 (EMPTY or DELETED) to 1111_1111
        // and high_bit = 0 (FULL) to 1000_0000
        //
        // Here's this logic expanded to concrete values:
        //   let special = 0 > tag = 1111_1111 (true) or 0000_0000 (false)
        //   1111_1111 | 1000_0000 = 1111_1111
        //   0000_0000 | 1000_0000 = 1000_0000
        unsafe {
            let special = neon::vcltzq_s8(neon::vreinterpretq_s8_u8(self.0));
            Group(neon::vorrq_u8(special, neon::vdupq_n_u8(0x80)))
        }
    }
}
