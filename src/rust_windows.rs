////////////  Prototype  ///////////
/*
 * Windows object to manage slicing and cellular
 * id of the frame buffer the idea is to provide
 * dynamic windows slicing and safe shared access
 * for fast processing.
 */

use rayon::prelude::*;
use std::mem::transmute;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub struct FrameBufferCell<T>
where
    T: Fn(Arc<[AtomicU32]>, usize, &(usize, usize), &(usize, usize)) + Send + Sync, // Closure must be thread-safe
{
    pub cell_id: usize,
    pub anchor_point: (usize, usize),
    pub frame_vector: (usize, usize),
    pub runtime: T, // Function defining per-cell behavior
}

/*
* Arc<[AtomicU32]> data type allows safe mutation across threads
* without mutex lock retention, improving performance.
*/

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FrameBufferStrategy<T>
where
    T: Fn(Arc<[AtomicU32]>, usize, &(usize, usize), &(usize, usize)) + Send + Sync + 'static, // Thread-safe closure
{
    // Shared frame buffer across all threads
    pub frame_buffer: Arc<[AtomicU32]>,
    // Minifb screen resolution
    pub global_screen_width: usize,
    pub global_screen_height: usize,
    // Frame cells to compute
    pub cells_stack: Vec<FrameBufferCell<T>>,
}

impl<T> FrameBufferStrategy<T>
where
    T: Fn(Arc<[AtomicU32]>, usize, &(usize, usize), &(usize, usize)) + Send + Sync + 'static, // Closure needs to be safely shared across threads
{
    ////////////////////////////////////////////////////////////////////////////
    // Use Rayon to parallelize buffer processing
    #[allow(dead_code)]
    pub fn compute_and_write_buffer_cells(&self) {
        self.cells_stack.par_iter().for_each(|cell| {
            let fb_clone = Arc::clone(&self.frame_buffer);
            // Run the cell's custom function
            (cell.runtime)(
                fb_clone,
                cell.cell_id,
                &cell.anchor_point,
                &cell.frame_vector,
            );
        });
    }
    ////////////////////////////////////////////////////////////////////////////
    // Convert AtomicU32 based buffer into Minifb standard.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn atomic_to_u32_safe(atomics: &[AtomicU32]) -> Vec<u32> {
        atomics.iter().map(|a| a.load(Ordering::Relaxed)).collect()
    }
    ////////////////////////////////////////////////////////////////////////////
    #[allow(dead_code)]
    #[inline(always)]
    pub fn atomic_to_u32_zero_copy(atomics: &[AtomicU32]) -> &[u32] {
        unsafe { transmute(atomics) }
    }
}
