% I'm hoping someone will find some use for this
% Author: Mindaugas Jarmolovičius

if ispc
	uv=['C:\\Users\\' getenv('USERNAME') '\\.local\\bin\\uv.exe'];
	if exist(uv, 'file')~=2
		system('powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"')
	end
else
	uv=[getenv('HOME') '/.local/bin/uv'];
	if exist(uv, 'file')~=2
		system('curl -LsSf https://astral.sh/uv/install.sh | sh')
	end
end

% Details at https://www.mathworks.com/support/requirements/python-compatibility.html
if isMATLABReleaseOlderThan("R2018a")
	error('Matlab release is too old!')
elseif isMATLABReleaseOlderThan("R2021b")
	warning("Old matlab version, expect problems with python")
	pythonVer='cpython-3.7';
elseif isMATLABReleaseOlderThan("R2022b")
	pythonVer='cpython-3.9';
elseif isMATLABReleaseOlderThan("R2026a")
	pythonVer='cpython-3.10';
else
	warning("New matlab version, add check for python version compatiblity")
	pythonVer='cpython-3.13';
end

dependencies = {
	'rich',  % just an example
};

depStr = strtrim(sprintf('%s,' ,dependencies{:}));
depStr = depStr(1:end-1);
disp("Setting up python..")
uvCmd = [uv ' ' 'tool run -qp ' pythonVer ' --with ' depStr ' python -c "import sys; print(sys.executable)"'];
[status, resp] = system(uvCmd);
if status~=0
	disp(["Executed: " uvCmd])
	disp(resp)
	error("Something went wrong with uv command!")
end

pe = pyenv(Version=strip(resp), ExecutionMode="OutOfProcess");
