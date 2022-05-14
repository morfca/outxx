package main

import "fmt"
import "sync"


func check(e error) {
	if e != nil {
		panic(e)
	}
}

type bitpat struct {
	pattern uint64
	length int
}

func testBitpatForMatch (v bitpat, matches *[32]int) (int) {
	matchcount := 0
	for i := 1; i <= v.length / 2; i++ {
		length_mask := (^uint64(0) >> (64 - i))
		a := v.pattern & length_mask
		b := (v.pattern >> i) & length_mask
		if a == b {
			matches[matchcount] = i * 2
			matchcount++
		}
	}
	return matchcount
}

func initBitpatFromInt (val uint64, l int) (bitpat) {
	if l > 64 {
		panic("greater than 64 bits unsupported")
	}
	return bitpat{pattern: (^uint64(0) >> (64 - l)) & val, length: l}
}


func initPrefixes(p2 int) []bool {
	if p2 > 32 {
		panic("value too high")
	}
	prefixes := make([]bool, 1 << p2)
	var matches [32]int
	for i := uint64(0); i < 1 << p2; i++ {
		pat := initBitpatFromInt(i, p2)
		nm := testBitpatForMatch(pat, &matches)
		if nm > 0 {
			prefixes[i] = true
		} else {
			prefixes[i] = false
		}
	}
	return prefixes
}

func invertBitpat(v bitpat) bitpat {
	return bitpat{
		pattern: ^v.pattern & (^uint64(0) >> (64 - v.length)),
		length: v.length,
	}
}

func bitpatToString(v bitpat) string {
	out := make([]byte, v.length)
	// fmt.Println(v.pattern, v.length)
	// fmt.Printf("%x\n", v.pattern)
	for i := 0; i < v.length; i++ {
		out[i] = byte(((v.pattern >> i) & 1) + 48)
	}
	return string(out)
}

func shard(init uint64, max uint64, l int, prefixes []bool) {
	pm := ^uint64(0) >> (64 - (l/2))
	var matches [32]int
	for i := init; i < max; i++ {
		if l >= 16 && prefixes[i & pm] {
			continue
		}
		pat := initBitpatFromInt(i, l)
		nm := testBitpatForMatch(pat, &matches)
		if nm > 0 {
			if nm >= 2 && matches[0] >= l/2 && matches[nm-1] == l && matches[0] >= matches[nm-1]/2 && matches[0] * 2 > matches[1] {
				fmt.Println(bitpatToString(pat), matches[0:nm])
				fmt.Println(bitpatToString(invertBitpat(pat)), matches[0:nm])
			}
		}
	}
}

func main() {
	for i := 2; i <= 24; i += 2 {
		prefixes := initPrefixes(i / 2)
		// -1 to eliminate half the run because it always produces bit-inverted duplciates
		max := uint64(1) << (i - 1)
		inc := uint64(1)
		if i >= 16 { inc = max >> 4 } else { inc = max }
		var wg sync.WaitGroup
		for j := uint64(0); j < max; j += inc {
			i := i
			j := j
			inc := inc
			wg.Add(1)
			func(){
				defer wg.Done()
				shard(j,j+inc,i, prefixes)
			}()
		}
		wg.Wait()
	}
}
