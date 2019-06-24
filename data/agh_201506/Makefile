dumps:
	mkdir -p dumps
	cd dumps; zstd -d --stdout ~/nfcaps.tar.zst | tar --strip-components 1 -xvf -

cleaned: dumps
	mkdir -p cleaned
	for file in dumps/2015-06-*/nfcapd.* ; \
	do \
	    nice nfdump -y -r $$file -w cleaned/$${file##*/} 'router ip 10.156.119.1 and if 83 and duration < 1000000' ; \
	done

merged: cleaned
	PYTHONPATH=../../../flow_models nice pypy3 -m flow_models.merge -i nfcapd -o binary -I 15 -A 300 -O merged cleaned

sorted: merged
	PYTHONPATH=../../../flow_models nice python3 -m flow_models.sort -k merged/last merged/last_ms -O sorted merged
