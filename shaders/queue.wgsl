// Based on https://github.com/GPUPeople/cuRE/blob/master/source/cure/pipeline/index_queue.cuh

requires unrestricted_pointer_parameters;

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

fn backoff() {
    // no op
}

fn size(q: ptr<storage, Queue, read_write>) -> i32 {
    return atomicLoad(&(*q).count);
}

fn ensure_enqueue(q: ptr<storage, Queue, read_write>) -> bool {
    if (atomicLoad(&(*q).count) >= i32(QUEUE_CAPACITY)) { return false; }

    let prev = atomicAdd(&(*q).count, 1);
    if (prev < i32(QUEUE_CAPACITY)) { return true; }
    atomicSub(&(*q).count, 1);
    return false;
}

fn ensure_dequeue(q: ptr<storage, Queue, read_write>) -> bool {
    if (atomicLoad(&(*q).count) <= 0) { return false; }

    let prev = atomicSub(&(*q).count, 1);
    if (prev > 0) { return true; }
    atomicAdd(&(*q).count, 1);
    return false;
}

fn publish_slot(q: ptr<storage, Queue, read_write>, p: u32, data: u32) {
    loop {
        let r = atomicCompareExchangeWeak(&(*q).ring[p], QUEUE_UNUSED, data);
        if (r.exchanged) { break; }
        backoff();
    }
}

fn consume_slot(q: ptr<storage, Queue, read_write>, p: u32) -> u32 {
    loop {
        let val = atomicExchange(&(*q).ring[p], QUEUE_UNUSED);
        if (val != QUEUE_UNUSED) {
            return val;
        }
        backoff();
    }
}

fn enqueue(q: ptr<storage, Queue, read_write>, data: u32) -> bool {
    if (data == QUEUE_UNUSED) { return false; }

    if (!ensure_enqueue(q)) { return false; }

    let pos = atomicAdd(&(*q).tail, 1u);
    let p = pos % QUEUE_CAPACITY;

    publish_slot(q, p, data);

    return true;
}

fn ensure_enqueue_n(q: ptr<storage, Queue, read_write>, count: u32) -> bool {
    let num = atomicLoad(&(*q).count);
    if (num + i32(count) > i32(QUEUE_CAPACITY)) { return false; }

    let prev = atomicAdd(&(*q).count, i32(count)); // old value
    if (prev + i32(count) <= i32(QUEUE_CAPACITY)) { return true; }
    atomicSub(&(*q).count, i32(count));
    return false;
}

fn enqueue_batch(q: ptr<storage, Queue, read_write>, items: array<u32, 4>, count: u32) -> bool {
    if (count == 0u) { return true; }
    if (count > 4u) { return false; }

    for (var i = 0u; i < count; i++) { if (items[i] == QUEUE_UNUSED) { return false; } }

    if (!ensure_enqueue_n(q, count)) { return false; }

    let start_pos = atomicAdd(&(*q).tail, count);

    for (var i = 0u; i < count; i++) {
        let val = items[i];
        let p = (start_pos + i) % QUEUE_CAPACITY;

        publish_slot(q, p, val);
    }

    return true;
}

fn ensure_dequeue_n(q: ptr<storage, Queue, read_write>, count: u32) -> bool {
    if (atomicLoad(&(*q).count) < i32(count)) { return false; }

    let prev = atomicSub(&(*q).count, i32(count));
    if (prev >= i32(count)) { return true; }
    atomicAdd(&(*q).count, i32(count));
    return false;
}

fn dequeue_batch(q: ptr<storage, Queue, read_write>, out_items: ptr<function, array<u32, 4>>, count: u32) -> bool {
    if (count > 4u) { return false; }
    if (!ensure_dequeue_n(q, count)) { return false; }

    let start_pos = atomicAdd(&(*q).head, count);
    
    for (var i = 0u; i < count; i++) {
        let p = (start_pos + i) % QUEUE_CAPACITY;

        (*out_items)[i] = consume_slot(q, p);
    }
    return true;
}

fn dequeue(q: ptr<storage, Queue, read_write>, out_data: ptr<function, u32>) -> bool {
    if (!ensure_dequeue(q)) { return false; }

    let pos = atomicAdd(&(*q).head, 1u);
    let p = pos % QUEUE_CAPACITY;

    (*out_data) = consume_slot(q, p);

    return true;
}
