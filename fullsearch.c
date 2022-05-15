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

// store the topmost lane from src to dest, then permute src in [a,b,c,d] => [b,c,d,a] order (57 is the magic number for that)
#define store_top_and_permute(src, dest, counter) _mm_storeu_si32((void *)&(dest[counter]), src); src = _mm_permute_ps(src, 57);

#define VECTOR_INCREMENT 4

int vectorized_test_mask(int i, bitpat_t p, int mc, int matches[]) {
	__m256i yi, ymask, ya, yb, yresult, ymatches, ymaskshift, ynotmask;
	__m128i xstore;
	uint32_t result_lanes;
	int j, ret;
	// initialize yi to the value of i and add offset vector {0, 1, 2, 3} to replace 4 loops on i
	// experimented with unrolling this loop, seems to make things worse on Zen 2.
	// probably because the execution path is only 256-bit wide on Zen 2
	// newer chips may yield better perf here.
	yi = _mm256_set1_epi64x(i);
	yi = _mm256_add_epi64(yi, offsets);
	// set a bitmask of the lower yi bits on ymask by initializing to all 1's, shfting left, then noting
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
	yresult = _mm256_cmpeq_epi64(ya, yb);
	result_lanes = _mm256_movemask_pd(yresult);
	// we want to store the size of the bit patterns that result in euqals, so...
	// reinitialize xstore to i and do offsets
	xstore = _mm_set1_epi32(i);
	xstore = _mm_add_epi32(xstore, offsets32);
	// multiply by 2 to get the bit pattern that doubles rather than the shift size for the compare
	xstore = _mm_sllv_epi32(xstore, xone);
	// use pre-calculated permute map to shuffle our results to the leftmost lanes
	xstore = _mm_permutevar_ps(xstore, collapse_map[result_lanes]);
	// then store to array.
	j = mc;
	// this is about a 3% speedup for the whole application vs an equivalent for loop ¯\_(ツ)_/¯
	switch (add_table[result_lanes]) {
		case 4: store_top_and_permute(xstore, matches, j++);
		case 3: store_top_and_permute(xstore, matches, j++);
		case 2: store_top_and_permute(xstore, matches, j++);
		case 1: store_top_and_permute(xstore, matches, j++);
		default: break;
	}
	return add_table[result_lanes];
}
#elif __ARM_NEON__
#include <arm_neon.h>
#define VECTOR_INCREMENT 2

uint64x2_t allmax = {UINT64_MAX, UINT64_MAX};
uint64x2_t voffsets = {0, 1};
uint64x2_t vone = {1, 1};
uint64x2_t vtwo = {2, 2};

inline int vectorized_test_mask(int i, bitpat_t p, int mc, int matches[]) {
	uint64x2_t vi, vold, vmask, va, vb, vresult, vguard;
	int temp, residx = mc;
	// we need the pattern length in vector form for later comparisons
	const uint64x2_t vplength = vdupq_n_u64(p.length);
	// initialize both lanes to i
	// increment the right lane by 1
	vi = vdupq_n_u64(i);
	vi = vaddq_u64(vi, voffsets);
	// create bitmask of i lower bits from vi lanes by initializing to all 1, shifting left by vi, then inverting
	vmask = vshlq_u64(allmax, vi);
	vmask = vmvnq_u8(vmask);
	// initialize a and b from pattern
	va = vb = vdupq_n_u64(p.pattern);
	va = vandq_u64(va, vmask);
	// shift vb left by -vi. we negate vi by bitwise not plus one
	vb = vshlq_u64(vb, vaddq_u64(vmvnq_u8(vi), vone));
	// apply bitmask to vb
	vb = vandq_u64(vb, vmask);
	// do comparison
	vresult = vceqq_u64(va, vb);
	// multiply by two to get the bit pattern that doubles rather than the shift size for the compare
	// use a bit shift because we know we are multiplying by 2 and the latency on bit shifts is less
	vold = vi;
	vi = vshlq_u64(vi, vone);
	// generate a bitmask using less than equal comparator to mask out values that go over the pattern length
	// vguard = vcleq_u64(vi, vplength);
	// store each lane if the compare lane shows a result
	// this is probably why it's barely faster than scalar but I can't figure out a better way to do this
	// vresult = vandq_u64(vresult, vguard);
	uint64_t res1 = vgetq_lane_u64(vresult, 0);
	uint64_t res2 = vgetq_lane_u64(vresult, 1);
	int resmsk = (res1 & 1) | ((res2 & 1) << 1);
	switch (resmsk) {
		case 3:
			matches[residx++] = (int) vgetq_lane_u64(vi, 0);
		case 2:
			vi = vextq_u64(vi, vi, 1);
		case 1:
			matches[residx++] = (int) vgetq_lane_u64(vi, 0);
		default: break;
	}
	return residx - mc;
}
#else
#define VECTOR_INCREMENT 1
#define vectorized_test_mask scalar_test_mask
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
	int i, ret, min, mc = 0;
	uint64_t a, b;
	min = i%4?i/4:(i/4)+1;
	for (i=min; i<=p.length/2; i += VECTOR_INCREMENT) {
		mc += vectorized_test_mask(i, p, mc, matches);
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

typedef struct {
	uint64_t jmax;
	uint64_t max;
	uint64_t j;
	int i;
	int prefix_guard;
	int inc;
	uint8_t *prefixes;
} item_t;

typedef struct {
	pthread_mutex_t mut;
	int size;
	int p, last;
	int done;
 	item_t item[1024];
} workqueue_t;

#define THREADS 8

void shard(item_t item) {
	int mc, mc2, matches[32+8], matches2[32+8];  // extra space in case of overflow from vector registers
	uint64_t mask;
	char pbuff[65];
	bitpat_t t;
	for (; item.j<item.jmax; item.j++) {
		mask = (uint64_t)(~0) >> (64 - (item.i/2));
		if (item.prefixes[item.j&mask]) {
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
		}
	}
}

#define POLL_INTERVAL 50

int shard_wrap(workqueue_t *wq) {
	int ret;
	item_t item;
	while (wq->done != 1) {
		ret = pthread_mutex_lock(&(wq->mut));
		if (ret) {
			fprintf(stderr, "mutex problem\n");
			return ret;
		}
		if (wq->last == wq->p) {
			pthread_mutex_unlock(&(wq->mut));
			usleep(POLL_INTERVAL);
			continue;
		}
		item = wq->item[wq->last++];
		if (wq->last >= wq->size) {
			wq->last = 0;
		}
		pthread_mutex_unlock(&(wq->mut));
		shard(item);
	}
	return 0;
}

int main() {
	int i, status, mc, outstanding, thr, ret;
	uint8_t *prefixes;
	uint64_t j, jmax, inc, max, mask;
	pid_t cp;
	pthread_t threads[1024];
	workqueue_t wq;
	item_t item;
	wq.size = 1024;
	wq.p = wq.last = 0;
	pthread_mutex_init(&(wq.mut), NULL);
	for (i=10; i<=32; i += 2) {
		wq.done = 0;
		max = (uint64_t)1 << (i - 1);
		prefixes = malloc((size_t) 1 << (i/2));
		init_prefixes(i/2, prefixes);
		inc = i>=16?max>>8:max;
		for (j=0; j<max;j+=inc) {
			jmax = j + inc;
			item.i = i; item.j = j; item.inc = inc; item.jmax = jmax; item.max = max; item.prefixes = prefixes;
			wq.item[wq.p++] = item;
			if (wq.p >= wq.size) {
				wq.p = 0;
			}
		}
		for (thr = 0; thr < THREADS; thr++) {
			ret = pthread_create(&(threads[thr]), NULL, (void *(*) (void *)) shard_wrap, &wq);
			if (ret) {
				fprintf(stderr, "thread problem\n");
				return 1;
			}
		}
		while (1) {
			usleep(POLL_INTERVAL);
			ret = pthread_mutex_lock(&wq.mut);
			if (ret) {
				fprintf(stderr, "mutex problem ret=%i\n", ret);
				return ret;
			}
			if (wq.p == wq.last) {
				wq.done = 1;
				pthread_mutex_unlock(&wq.mut);
				break;
			}
			pthread_mutex_unlock(&wq.mut);
		}
		for (thr = 0; thr < THREADS; thr++) {
			pthread_join(threads[thr], NULL);
		}
		free(prefixes);
	}
}