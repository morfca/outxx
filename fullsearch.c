#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


typedef struct {
	uint64_t pattern;
	int length;
} bitpat_t;

bitpat_t int2bitpat(uint64_t v, int l) {
	return (bitpat_t){~0 >> (64 - l) & v, l};
}

#ifdef __AVX2__
#include <immintrin.h>
#define m128_init(c1, c2, c3, c4) {(uint64_t)c1 + ((uint64_t)c2 << 32), (uint64_t)c3 + ((uint64_t)c4 << 32)}
#define m128_unpack(x) (int[4]){x[0]&0xFFFFFFFF, x[0]>>32, x[1]&0xFFFFFFFF, x[1]>>32}

__m256i offsets = { (uint64_t)0, (uint64_t)1, (uint64_t)2, (uint64_t)3 };
__m128i offsets32 = m128_init(0, 1, 2, 3);
__m128i xone = m128_init(1,1,1,1);

uint8_t add_table[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

// permute values that consolidate results in leftmost lanes
// we precompute this so we can quickly convert the result from cmpeq to an argument for permutevar
__m128i collapse_map[] = {
	m128_init(0,0,0,0),  // 0b0000
	m128_init(0,0,0,0),  // 0b0001
	m128_init(1,0,0,0),  // 0b0010
	m128_init(0,1,0,0),  // 0b0011
	m128_init(2,0,0,0),  // 0b0100
	m128_init(0,2,0,0),  // 0b0101
	m128_init(1,2,0,0),  // 0b0110
	m128_init(0,1,2,0),  // 0b0111
	m128_init(3,0,0,0),  // 0b1000
	m128_init(0,3,0,0),  // 0b1001
	m128_init(1,3,0,0),  // 0b1010
	m128_init(0,1,3,0),  // 0b1011
	m128_init(2,3,0,0),  // 0b1100
	m128_init(0,2,3,0),  // 0b1101
	m128_init(1,2,3,0),  // 0b1110
	m128_init(0,1,2,3),  // 0b1111
};

// try doing a table instead of broadcast


// store the topmost lane from src to dest, then permute src in [a,b,c,d] => [b,c,d,a] order (57 is the magic number for that)
#define store_leftmost(src, dest, counter) _mm_storeu_si32((void *)&(dest[counter]), src)
#define permute_left(src)  src = _mm_permute_ps(src, 57)

#define VECTOR_INCREMENT 4

int vectorized_test_mask(int i, bitpat_t p, int mc, int max, int matches[]) {
	__m256i yi, ymask, ya, yb, yresult, ymatches, ymaskshift, ynotmask, yguard;
	__m128i xstore, xpermute;
	uint32_t result_lanes;
	int ret, j = mc;
	// initialize yi to the value of i and add offset vector {0, 1, 2, 3} to replace 4 loops on i
	// experimented with unrolling this loop, seems to make things worse on Zen 2.
	// probably because the execution path is only 256-bit wide on Zen 2
	// newer chips may yield better perf here.
	yi = _mm256_set1_epi64x(i);
	yi = _mm256_add_epi64(yi, offsets);
	// generate mask to mask out any values of i that exceed max
	yguard = _mm256_set1_epi64x(max);
	yguard = _mm256_cmpgt_epi64(yguard, yi);
	// reinitialize xstore to i and do offsets
	xstore = _mm_set1_epi32(i);
	xstore = _mm_add_epi32(xstore, offsets32);
	// set a bitmask of the lower yi bits on ymask by initializing to all 1's, shfting left, then bitwise not
	ymask = ynotmask = _mm256_set1_epi64x(UINT64_MAX);
	ymask = _mm256_sllv_epi64(ymask, yi);
	ymask = _mm256_andnot_si256(ymask, ynotmask);
	ya = yb = _mm256_set1_epi64x(p.pattern);
	// mask out high bits on ya
	ya = _mm256_and_si256(ya, ymask);
	// shift yb right by yi then mask out high bits
	yb = _mm256_srlv_epi64(yb, yi);
	yb = _mm256_and_si256(yb, ymask);
	// compare, then get result lane mask
	yresult = _mm256_and_si256(_mm256_cmpeq_epi64(ya, yb), yguard);
	result_lanes = _mm256_movemask_pd(yresult);
	// do table lookup for precomputed permute indices
	xpermute = collapse_map[result_lanes];
	// we want to store the size of the bit patterns that result in euqals, so...
	// multiply by 2 to get the bit pattern that doubles rather than the shift size for the compare
	xstore = _mm_sllv_epi32(xstore, xone);
	// use pre-calculated permute map to shuffle our results to the leftmost lanes
	xstore = _mm_permutevar_ps(xstore, xpermute);
	// then store to array.
	// this is about a 3% speedup for the whole application vs an equivalent for loop ¯\_(ツ)_/¯
	switch (add_table[result_lanes]) {
		case 4: store_leftmost(xstore, matches, j++); permute_left(xstore);
		case 3: store_leftmost(xstore, matches, j++); permute_left(xstore);
		case 2: store_leftmost(xstore, matches, j++); permute_left(xstore);
		case 1: store_leftmost(xstore, matches, j++);
		case 0: {};
	}
	return j - mc;
}
#elif __SSE4_2__
#include <immintrin.h>
#include <emmintrin.h>
#define m128_init32(c1, c2, c3, c4) {(uint64_t)c1 + ((uint64_t)c2 << 32), (uint64_t)c3 + ((uint64_t)c4 << 32)}
#define m128_unpack32(x) (int[4]){x[0]&0xFFFFFFFF, x[0]>>32, x[1]&0xFFFFFFFF, x[1]>>32}
#define m128_init64(c1, c2) {(uint64_t)c1, (uint64_t)c2}
#define m128_unpack64(x) (uint64_t[2]){x[0], x[1]}

__m128i offsets32 = m128_init32(0, 1, 2, 3);
__m128i offsets64 = m128_init64(0, 1);
__m128i xone32 = m128_init32(1,1,1,1);
__m128i xone64 = m128_init64(1,1);

uint8_t add_table[] = {0, 1, 1, 2};

// # define store_top(src, dest, counter) _mm_storeu_si32((void *)&(dest[counter], src);

#define VECTOR_INCREMENT 2
int vectorized_test_mask(int i, bitpat_t p, int mc, int max, int matches[]) {
	__m128i xi, xmask, xa, xb, xresult, xmatches, xmaskshift, xnotmask;
	__m128i xstore;
	uint64_t mask, itemp;
	uint32_t result_lanes;
	int j, ret;
	xi = _mm_set1_epi64x(i);
	xstore = xi = _mm_add_epi64(xi, offsets64);
	// set a bitmask of the lower xi bits on ymask by initializing to all 1's, shfting left, then bitwise not
	mask = UINT64_MAX << i;
	// unfortunately no var shift in SSE :(, have to do the shift in scalar space
	xmask = _mm_insert_epi64(xmask, mask, 0);
	mask = mask << 1;
	xmask = _mm_insert_epi64(xmask, mask, 1);
	// AFAICT comparing something to itself is the fastest way to set all bits
	xnotmask = _mm_cmpeq_epi64(xi, xi);
	xmask = _mm_andnot_si128(xmask, xnotmask);
	xa = _mm_set1_epi64x(p.pattern);
	// mask out high bits on xa
	xa = _mm_and_si128(xa, xmask);
	// shift xb right by xi then mask out high bits
	itemp = p.pattern >> i;
	xb = _mm_insert_epi64(xb, itemp, 0);
	itemp = itemp >> 1;
	xb = _mm_insert_epi64(xb, itemp, 1);
	xb = _mm_and_si128(xb, xmask);
	// compare, then get result lane mask
	xresult = _mm_cmpeq_epi64(xa, xb);
	result_lanes = _mm_movemask_pd(xresult);
	// we want to store the size of the bit patterns that result in equals, so...
	// multiply by 2 to get the bit pattern that doubles rather than the shift size for the compare
	xstore = _mm_sll_epi64(xstore, xone64);
	// use pre-calculated permute map to shuffle our results to the leftmost lanes
	j = mc;
	xstore = _mm_shuffle_ps(xstore, xstore, 8);
	switch(result_lanes) {
		case 3: _mm_storeu_si32((void *)&(matches[j++]), xstore);
		case 2: xstore = _mm_shuffle_ps(xstore, xstore, 1);
		case 1: _mm_storeu_si32((void *)&(matches[j++]), xstore);
		case 0: {};
	}
	return j - mc;
}
#elif __ARM_NEON__
#include <arm_neon.h>
#define VECTOR_INCREMENT 4
uint64x2_t allmax = {UINT64_MAX, UINT64_MAX};
uint64x2_t voffsets01 = {0, 1};
uint64x2_t voffsets23 = {2, 3};
uint64x2_t vone = {1, 1};
uint64x2_t vtwo = {2, 2};

int vectorized_test_mask(int i, bitpat_t p, int mc, int max, int matches[]) {
	uint64x2_t vi1, vmask1, va1, vb1, vresult1, vguard1;
	uint64x2_t vi2, vmask2, va2, vb2, vresult2, vguard2;
	int temp, j = mc, resmask = 0;
	uint64_t res;
	// we need the pattern length in vector form for later comparisons
	const uint64x2_t vplength = vdupq_n_u64(p.length);
	// initialize both lanes to i
	// increment the right lane by 1
	vi1 = vdupq_n_u64(i);
	vi1 = vaddq_u64(vi1, voffsets01);
	vi2 = vdupq_n_u64(i);
	vi2 = vaddq_u64(vi2, voffsets23);
	// compute guard mask
	vguard1 = vcleq_u64(vi1, vdupq_n_u64(max));
	vguard2 = vcleq_u64(vi2, vdupq_n_u64(max));
	// create bitmask of i lower bits from vi lanes by initializing to all 1,
	// shifting left by vi, then inverting
	vmask1 = vshlq_u64(allmax, vi1);
	vmask1 = vmvnq_u8(vmask1);
	vmask2 = vshlq_u64(allmax, vi2);
	vmask2 = vmvnq_u8(vmask2);
	// initialize a and b from pattern
	va1 = vb1 = vdupq_n_u64(p.pattern);
	va1 = vandq_u64(va1, vmask1);
	va2 = vb2 = vdupq_n_u64(p.pattern);
	va2 = vandq_u64(va2, vmask2);
	// shift vb left by -vi. we negate vi by bitwise not plus one
	vb1 = vshlq_u64(vb1, vaddq_u64(vmvnq_u8(vi1), vone));
	vb2 = vshlq_u64(vb2, vaddq_u64(vmvnq_u8(vi2), vone));
	// apply bitmask to vb
	vb1 = vandq_u64(vb1, vmask1);
	vb2 = vandq_u64(vb2, vmask2);
	// do comparison
	vresult1 = vceqq_u64(va1, vb1);
	vresult1 = vandq_u64(vresult1, vguard1);
	vresult2 = vceqq_u64(va2, vb2);
	vresult2 = vandq_u64(vresult2, vguard2);
	// multiply by two to get the bit pattern that doubles rather than the shift size for the compare
	// use a bit shift because we know we are multiplying by 2 and the latency on bit shifts is less
	// get the lowest bit from each lane
	vresult1 = vandq_u64(vresult1, vone);
	vresult2 = vandq_u64(vresult2, vone);
	// shift each lane left by a different amount so they're in the lowest 4 bits
	vresult1 = vshlq_u64(vresult1, voffsets01);
	vresult2 = vshlq_u64(vresult2, voffsets23);
	// we want to OR all lanes in both results into a single scalar vector, so...
	// OR the two vectors together
	vresult1 = vorrq_u64(vresult1, vresult2);
	// does not appear to be an cross-lane OR intrinsic but add should be
	// equivalent because we shifted each lane by a different amount
	resmask = vpaddd_u64(vresult1);
	for (i <<= 1; resmask; resmask >>= 1, i += 2) {
		if (resmask & 1) matches[j++] = i;
	}
	return j - mc;
}
#else
#define VECTOR_INCREMENT 1
#define vectorized_test_mask(a, b, c, d, e) scalar_test_mask(a, b, c, e)
#endif

int scalar_test_mask(int i, bitpat_t p, int mc, int matches[]) {
	uint64_t mask = (uint64_t)(~0) >> (64 - i);  // cast to avoid sign extension
	uint64_t a = p.pattern & mask;
	uint64_t b = (p.pattern >> i) & mask;
	int mc_inc = 0;
	if (a==b) {
		matches[mc+mc_inc++] = i*2;
	}
	return mc_inc;
}

int simd_test_for_match(bitpat_t p, int matches[]) {
	int i, ret, min, max, mc = 0;
	uint64_t a, b;
	min = i%4?i/4:(i/4)+1;
	max = p.length/2+1;
	for (i=min; i<=max; i += VECTOR_INCREMENT) {
		mc += vectorized_test_mask(i, p, mc, max, matches);
	}
	return mc>2?2:mc;
}

int test_for_match(bitpat_t p, int matches[]) {
	int i, min, mc = 0;
	uint64_t a, b, mask;
	min = i%4?i/4:(i/4)+1;
	for (i=min; i<=p.length/2; i++) {
		mc += scalar_test_mask(i, p, mc, matches);
	}
	return mc;
}

int bitpat2str(bitpat_t p, char *buff) {
	int i;
	for (i=0; i<p.length; i++) {
		buff[i] = ((p.pattern >> i) & 1) + 48;
		buff[i+1] = '\0';
	}
	return 1;
}

int init_prefixes(int p2, uint8_t *prefixes) {
	int i, matches[32];
	bitpat_t t;
	for (i=0; i < 1 << p2; i++) {
		t = int2bitpat(i, p2);
		prefixes[i] = (uint8_t) test_for_match(t, matches);
	}
	return 1;
}

#define QUEUE_DEPTH 16

typedef struct {
	int refcount;
	uint8_t *prefixes;
} prefix_cache_t;

typedef struct {
	uint64_t jmax;
	uint64_t max;
	uint64_t j;
	int i;
	int prefix_guard;
	int inc;
	prefix_cache_t *prefixes;
} item_t;

typedef struct {
	pthread_mutex_t mut;
	int size;
	int p, last;
	int count;
	int done;
	item_t item[1 << QUEUE_DEPTH];
} workqueue_t;

#define THREADS 4

void shard(item_t item) {
	int mc, mc2, matches[32+8], matches2[32+8];  // extra space in case of overflow from vector registers
	uint64_t mask;
	char pbuff[65];
	bitpat_t t;
	for (; item.j<item.jmax; item.j++) {
		mask = (uint64_t)(~0) >> (64 - (item.i/2));
		if (item.prefixes->prefixes[item.j&mask]) {
			continue;
		}
		t = int2bitpat(item.j,item.i);
		mc = simd_test_for_match(t, matches);
		if (mc == 2 && matches[0] >= item.i/2 && matches[1] == item.i && matches[0] >= matches[1]/2 && matches[0]*2 > matches[1]) {
			bitpat2str(t, pbuff);
			printf("%s [%i %i]\n", pbuff, matches[0], matches[1]);
			t.pattern = ~t.pattern;
			bitpat2str(t, pbuff);
			printf("%s [%i %i]\n", pbuff, matches[0], matches[1]);
			fflush(stdout);
		}
	}
}

#define POLL_INTERVAL 100

int shard_wrap(workqueue_t *wq) {
	int ret;
	item_t item;
	while (wq->done != 1) {
		ret = pthread_mutex_lock(&(wq->mut));
		if (ret) {
			fprintf(stderr, "mutex problem (dequeue) %i\n", ret);
			exit(1);
		}
		if (wq->count == 0) {
			pthread_mutex_unlock(&(wq->mut));
			usleep(POLL_INTERVAL);
			continue;
		}
		wq->count--;
		item = wq->item[wq->last++];
		if (wq->last >= wq->size) {
			wq->last = 0;
		}
		pthread_mutex_unlock(&(wq->mut));
		shard(item);
		ret = pthread_mutex_lock(&(wq->mut));
		if (ret) {
			fprintf(stderr, "mutex problem (cleanup) %i\n", ret);
			exit(1);
		}
		item.prefixes->refcount--;
		if (item.prefixes->refcount == 0) {
			free(item.prefixes->prefixes);
			free(item.prefixes);
		}
		pthread_mutex_unlock(&(wq->mut));
	}
	return 0;
}

#define PLIMIT 32

int main() {
	int i, k, status, mc, outstanding, thr, ret, shift, curcount;
	uint64_t j, jmax, inc, max, mask;
	pid_t cp;
	pthread_t threads[THREADS];
	prefix_cache_t *prefixes;
	workqueue_t wq;
	item_t item;
	wq.size = 1 << QUEUE_DEPTH;
	wq.p = wq.last = 0;
	wq.done = 0;
	pthread_mutex_init(&(wq.mut), NULL);
	for (thr = 0; thr < THREADS; thr++) {
		ret = pthread_create(&(threads[thr]), NULL, (void *(*) (void *)) shard_wrap, &wq);
		if (ret) return 1;
	}
	for (i = 10; i <= PLIMIT; i += 2) {
		max = (uint64_t)1 << (i - 1);
		inc = i >= 16 ? max >> (i / 4) : max;
		prefixes = malloc(sizeof(prefix_cache_t));
		prefixes->prefixes = malloc((size_t) 1 << (i/2));
		init_prefixes(i/2, prefixes->prefixes);
		for (j = 0; j < max;) {
			while (1) {
				ret = pthread_mutex_lock(&wq.mut);
				if (ret) {
					fprintf(stderr, "mutex problem (insert) %i\n", ret);
					return 1;
				}
				if (wq.count < (1 << (QUEUE_DEPTH - 1))) {
					break;
				}
				pthread_mutex_unlock(&wq.mut);
				usleep(POLL_INTERVAL * 5);
			}
			for (k = 0; (j < max) && (k < (1 << (QUEUE_DEPTH - 2))); j += inc, k++) {
				jmax = j + inc;
				item.i = i; item.j = j; item.inc = inc; item.jmax = jmax; item.max = max; item.prefixes = prefixes;
				prefixes->refcount++;
				wq.item[wq.p++] = item;
				wq.count++;
				wq.p = wq.p < wq.size ? wq.p : 0;
			}
			pthread_mutex_unlock(&wq.mut);
		}
	}
	while (1) {
		ret = pthread_mutex_lock(&wq.mut);
		if (ret) {
			fprintf(stderr, "mutex problem (end) %i\n", ret);
			return 1;
		}
		if (wq.p == wq.last) {
			wq.done = 1;
			pthread_mutex_unlock(&wq.mut);
			break;
		}
		pthread_mutex_unlock(&wq.mut);
		usleep(POLL_INTERVAL);
	}
	for (thr = 0; thr < THREADS; thr++) {
		pthread_join(threads[thr], NULL);
	}
	printf("done\n");
}
