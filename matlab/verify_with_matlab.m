function verify_with_matlab(vector_dir)
%VERIFY_WITH_MATLAB  Independent check of the C++ scrambler against 5G Toolbox.
%
%   Every vector in test/vectors is re-scrambled with nrPUSCHScramble and
%   compared against the committed .ref file, so the C++ implementation, the
%   Python spec reference and MathWorks' model must all agree bit for bit.
%
%   Usage:
%       verify_with_matlab                 % uses ../test/vectors
%       verify_with_matlab('path/to/dir')
%
%   Requires: 5G Toolbox (nrPUSCHScramble).

    if nargin < 1
        vector_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'test', 'vectors');
    end

    if isempty(which('nrPUSCHScramble'))
        error('verify:no5G', '5G Toolbox is required (nrPUSCHScramble not found).');
    end

    files = dir(fullfile(vector_dir, '*.in'));
    if isempty(files)
        error('verify:noVectors', 'No .in vectors found in %s', vector_dir);
    end

    fprintf('%-14s %8s %6s %8s   %s\n', 'vector', 'rnti', 'nid', 'bits', 'result');
    fprintf('%s\n', repmat('-', 1, 52));

    failures = 0;
    for k = 1:numel(files)
        name = files(k).name;
        stem = name(1:end-3);
        in_path  = fullfile(vector_dir, name);
        ref_path = fullfile(vector_dir, [stem '.ref']);
        if ~isfile(ref_path)
            fprintf('%-14s %8s %6s %8s   SKIP (no .ref)\n', stem, '-', '-', '-');
            continue;
        end

        [in_bytes, n_bits, rnti, nid] = read_vector(in_path);
        ref_bytes = read_vector(ref_path);

        in_bits  = unpack_lsb(in_bytes, n_bits);
        out_bits = nrPUSCHScramble(double(in_bits), nid, rnti);
        got      = pack_lsb(uint8(out_bits(:)));
        expected = ref_bytes(1:numel(got));

        if isequal(got, expected)
            verdict = 'PASS';
        else
            verdict = sprintf('FAIL (%d differing bytes)', sum(got ~= expected));
            failures = failures + 1;
        end
        fprintf('%-14s %8d %6d %8d   %s\n', stem, rnti, nid, n_bits, verdict);
    end

    fprintf('%s\n', repmat('-', 1, 52));
    if failures == 0
        fprintf('All %d vectors match nrPUSCHScramble.\n', numel(files));
    else
        error('verify:mismatch', '%d vector(s) did not match.', failures);
    end
end

% -------------------------------------------------------------------------

function [bytes, n_bits, rnti, nid] = read_vector(path)
%READ_VECTOR  Parse the hex-byte format: '#' comments, optional '# bits=N',
%             '# rnti=N nid=N' header on input files.
    txt = fileread(path);
    lines = regexp(txt, '\r\n|\r|\n', 'split');

    n_bits = 0; rnti = 0; nid = 0;
    tokens = {};
    for i = 1:numel(lines)
        line = lines{i};
        hash = strfind(line, '#');
        if ~isempty(hash)
            comment = line(hash(1):end);
            line = line(1:hash(1)-1);
            n_bits = scan_key(comment, 'bits=',  n_bits);
            rnti   = scan_key(comment, 'rnti=',  rnti);
            nid    = scan_key(comment, 'nid=',   nid);
        end
        parts = regexp(strtrim(line), '\s+', 'split');
        tokens = [tokens, parts(~cellfun('isempty', parts))]; %#ok<AGROW>
    end

    bytes = uint8(cellfun(@(s) hex2dec(s), tokens(:)));
    if n_bits == 0
        n_bits = numel(bytes) * 8;
    end
end

function v = scan_key(comment, key, v)
    idx = strfind(comment, key);
    if ~isempty(idx)
        num = sscanf(comment(idx(1)+numel(key):end), '%d', 1);
        if ~isempty(num)
            v = num;
        end
    end
end

function bits = unpack_lsb(bytes, n_bits)
%UNPACK_LSB  bit i of the stream lives at bit (i mod 8) of byte floor(i/8).
    bits = zeros(n_bits, 1, 'uint8');
    for i = 1:n_bits
        b = floor((i - 1) / 8) + 1;
        bits(i) = bitand(bitshift(bytes(b), -mod(i - 1, 8)), 1);
    end
end

function bytes = pack_lsb(bits)
    n = numel(bits);
    bytes = zeros(ceil(n / 8), 1, 'uint8');
    for i = 1:n
        if bits(i)
            b = floor((i - 1) / 8) + 1;
            bytes(b) = bitor(bytes(b), bitshift(uint8(1), mod(i - 1, 8)));
        end
    end
end
