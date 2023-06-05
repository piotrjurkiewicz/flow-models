export PYTHONPATH = $(CURDIR)/../..
.SECONDARY:

dumps:
	mkdir -p dumps
	cd dumps; zstd -d --stdout ~/nfcaps.tar.zst | tar -xv

cleaned: dumps
	mkdir -p cleaned
	for file in dumps/nfcapd.* ; \
	do \
	    nice nfdump -y -r $$file -w cleaned/$${file##*/} 'router ip 10.156.119.1 and if 83 and duration < 1000000' ; \
	done

merged: cleaned
	nice pypy3 -m flow_models.merge -i nfcapd -o binary -I 15 -A 300 -O merged cleaned

sorted: merged
	nice python3 -m flow_models.sort -k last last_ms -O sorted merged

histograms/%/length.csv: sorted
	mkdir -p $(@D)
	nice pypy3 -m flow_models.hist -i binary -x length -b 12 --prot $* sorted > $@

histograms/%/length_b0.csv: sorted
	mkdir -p $(@D)
	nice pypy3 -m flow_models.hist -i binary -x length -b 0 --prot $* sorted > $@

histograms/%/size.csv: sorted
	mkdir -p $(@D)
	nice pypy3 -m flow_models.hist -i binary -x size -b 12 --prot $* sorted > $@

histograms/%/size_b0.csv: sorted
	mkdir -p $(@D)
	nice pypy3 -m flow_models.hist -i binary -x size -b 0 --prot $* sorted > $@

.SECONDEXPANSION:

histograms/%: histograms/$$*/length.csv histograms/$$*/length_b0.csv histograms/$$*/size.csv histograms/$$*/size_b0.csv
	true

mixtures/all/length: histograms/all/length.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.fit -i 400 -U 6 -L 4 -y flows ../../../histograms/all/length.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 6 -L 4 -y packets ../../../histograms/all/length.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 6 -L 4 -y octets ../../../histograms/all/length.csv
	touch $@

mixtures/all/size: histograms/all/size.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.fit -i 400 -U 0 -L 8 -y flows ../../../histograms/all/size.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 0 -L 5 -y packets ../../../histograms/all/size.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 0 -L 5 -y octets ../../../histograms/all/size.csv
	touch $@

mixtures/tcp/length: histograms/tcp/length.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.fit -i 500 -U 6 -L 4 -y flows ../../../histograms/tcp/length.csv
	cd $@; python3 -m flow_models.fit -i 500 -U 6 -L 4 -y packets ../../../histograms/tcp/length.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 2 -L 4 -y octets ../../../histograms/tcp/length.csv
	touch $@

mixtures/tcp/size: histograms/tcp/size.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.fit -i 400 -U 0 -L 8 -y flows ../../../histograms/tcp/size.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 0 -L 5 -y packets ../../../histograms/tcp/size.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 0 -L 5 -y octets ../../../histograms/tcp/size.csv
	touch $@

mixtures/udp/length: histograms/udp/length.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.fit -i 400 -U 4 -L 9 -y flows ../../../histograms/udp/length.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 7 -L 5 -y packets ../../../histograms/udp/length.csv
	cd $@; python3 -m flow_models.fit -i 400 -U 7 -L 5 -y octets ../../../histograms/udp/length.csv
	touch $@

mixtures/udp/size: histograms/udp/size.csv
	mkdir -p $@
	cd $@; python3 -m flow_models.fit -i 200 -U 0 -L 9 -y flows ../../../histograms/udp/size.csv
	cd $@; python3 -m flow_models.fit -i 200 -U 0 -L 9 -y packets ../../../histograms/udp/size.csv
	cd $@; python3 -m flow_models.fit -i 200 -U 0 -L 9 -y octets ../../../histograms/udp/size.csv
	touch $@

mixtures/%: mixtures/$$*/length mixtures/$$*/size
	true

plots/%: histograms/$$*/length.csv histograms/$$*/size.csv mixtures/$$*/length mixtures/$$*/size
	mkdir -p $@/length
	cd $@/length; python3 -m flow_models.plot -P points comp --format pdf -x length ../../../histograms/$*/length.csv ../../../mixtures/$*/length
	cd $@/length; python3 -m flow_models.plot -P points comp --format pdf --single -x length ../../../histograms/$*/length.csv ../../../mixtures/$*/length
	mkdir -p $@/size
	cd $@/size; python3 -m flow_models.plot -P points comp --format pdf -x size ../../../histograms/$*/size.csv ../../../mixtures/$*/size
	cd $@/size; python3 -m flow_models.plot -P points comp --format pdf --single -x size ../../../histograms/$*/size.csv ../../../mixtures/$*/size

summary/%: histograms/$$*/length_b0.csv histograms/$$*/size_b0.csv
	python3 -m flow_models.summary histograms/$*/length_b0.csv
	python3 -m flow_models.summary histograms/$*/size_b0.csv
