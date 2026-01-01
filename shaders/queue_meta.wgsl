const QUEUE_CAPACITY: u32 = 1024 * 1024 * 32;
const QUEUE_UNUSED: u32 = 0xFFFFFFFFu;

struct Queue {
    pad1: array<u32, 31>,
    head: atomic<u32>,

    pad2: array<u32, 31>,
    tail: atomic<u32>,

    pad3: array<u32, 31>,
    count: atomic<i32>,

    ring: array<atomic<u32>>,
}

struct QueueNonatomic {
    pad1: array<u32, 31>,
    head: u32,

    pad2: array<u32, 31>,
    tail: u32,
    
    pad3: array<u32, 31>,
    count: i32,

    ring: array<u32>,
}