// Based on https://github.com/GPUPeople/cuRE/blob/master/source/cure/pipeline/index_queue.cuh

#import "queue_meta.wgsl"

fn size() -> i32 {
    return atomicLoad(&queue.count);
}

fn ensure_enqueue() -> bool {
    if (atomicLoad(&queue.count) >= i32(QUEUE_CAPACITY)) { return false; }

    let prev = atomicAdd(&queue.count, 1);
    if (prev < i32(QUEUE_CAPACITY)) { return true; }
    atomicSub(&queue.count, 1);
    return false;
}

fn ensure_dequeue() -> bool {
    if (atomicLoad(&queue.count) <= 0) { return false; }

    let prev = atomicSub(&queue.count, 1);
    if (prev > 0) { return true; }
    atomicAdd(&queue.count, 1);
    return false;
}

fn publish_slot(p: u32, data: u32) {
    loop {
        let r = atomicCompareExchangeWeak(&queue.ring[p], QUEUE_UNUSED, data);
        if (r.exchanged) { break; }
    }
}

fn consume_slot(p: u32) -> u32 {
    loop {
        let val = atomicExchange(&queue.ring[p], QUEUE_UNUSED);
        if (val != QUEUE_UNUSED) {
            return val;
        }
    }
}

fn enqueue(data: u32) -> bool {
    if (data == QUEUE_UNUSED) { return false; }

    if (!ensure_enqueue()) { return false; }

    let pos = atomicAdd(&queue.tail, 1u);
    let p = pos % QUEUE_CAPACITY;

    publish_slot(p, data);

    return true;
}

fn dequeue(out_data: ptr<function, u32>) -> bool {
    if (!ensure_dequeue()) { return false; }

    let pos = atomicAdd(&queue.head, 1u);
    let p = pos % QUEUE_CAPACITY;

    (*out_data) = consume_slot(p);

    return true;
}

fn ensure_enqueue_n(count: u32) -> bool {
    let num = atomicLoad(&queue.count);
    if (num + i32(count) > i32(QUEUE_CAPACITY)) { return false; }

    let prev = atomicAdd(&queue.count, i32(count));
    if (prev + i32(count) <= i32(QUEUE_CAPACITY)) { return true; }
    atomicSub(&queue.count, i32(count));
    return false;
}

fn ensure_dequeue_n(count: u32) -> bool {
    if (atomicLoad(&queue.count) < i32(count)) { return false; }

    let prev = atomicSub(&queue.count, i32(count));
    if (prev >= i32(count)) { return true; }
    atomicAdd(&queue.count, i32(count));
    return false;
}

struct WgDequeueResult {
    start: u32,
    count: u32,
}

fn wg_dequeue_reserve(max_count: u32) -> WgDequeueResult {
    var result: WgDequeueResult;
    
    let available = atomicLoad(&queue.count);
    if (available <= 0) { return result; }
    
    let to_take = min(u32(available), max_count);
    
    let prev = atomicSub(&queue.count, i32(to_take));
    if (prev >= i32(to_take)) {
        result.start = atomicAdd(&queue.head, to_take);
        result.count = to_take;
    } else {
        atomicAdd(&queue.count, i32(to_take));
    }
    
    return result;
}

fn wg_dequeue_consume(start: u32, offset: u32) -> u32 {
    let p = (start + offset) % QUEUE_CAPACITY;
    return consume_slot(p);
}

fn wg_enqueue_reserve(count: u32) -> u32 {
    if (count == 0u) { return 0u; }
    
    loop {
        if (ensure_enqueue_n(count)) {
            return atomicAdd(&queue.tail, count);
        }
    }
}

fn wg_enqueue_publish(start: u32, offset: u32, data: u32) {
    let p = (start + offset) % QUEUE_CAPACITY;
    publish_slot(p, data);
}
