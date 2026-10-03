# Requires ffmpeg in PATH.
# Put this script next to ali-moshavere.m4a and run from PowerShell.

$input = "ali-moshavere.m4a"

$chunks = @(
  @{n="01"; start="00:00:00"; dur="00:10:00"; out="ali-moshavere_part01_00-00-00_to_00-10-00.m4a"},
  @{n="02"; start="00:09:57"; dur="00:10:00"; out="ali-moshavere_part02_00-09-57_to_00-19-57.m4a"},
  @{n="03"; start="00:19:54"; dur="00:10:00"; out="ali-moshavere_part03_00-19-54_to_00-29-54.m4a"},
  @{n="04"; start="00:29:51"; dur="00:10:00"; out="ali-moshavere_part04_00-29-51_to_00-39-51.m4a"},
  @{n="05"; start="00:39:48"; dur="00:10:00"; out="ali-moshavere_part05_00-39-48_to_00-49-48.m4a"},
  @{n="06"; start="00:49:45"; dur="00:10:00"; out="ali-moshavere_part06_00-49-45_to_00-59-45.m4a"},
  @{n="07"; start="00:59:42"; dur="00:10:00"; out="ali-moshavere_part07_00-59-42_to_01-09-42.m4a"},
  @{n="08"; start="01:09:39"; dur="00:10:00"; out="ali-moshavere_part08_01-09-39_to_01-19-39.m4a"},
  @{n="09"; start="01:19:36"; dur="00:10:00"; out="ali-moshavere_part09_01-19-36_to_01-29-36.m4a"},
  @{n="10"; start="01:29:33"; dur="00:10:00"; out="ali-moshavere_part10_01-29-33_to_01-39-33.m4a"},
  @{n="11"; start="01:39:30"; dur="00:10:00"; out="ali-moshavere_part11_01-39-30_to_01-49-30.m4a"},
  @{n="12"; start="01:49:27"; dur="00:09:50"; out="ali-moshavere_part12_01-49-27_to_01-59-17.m4a"}
)

foreach ($c in $chunks) {
  Write-Host "Creating $($c.out)..."
  ffmpeg -hide_banner -loglevel warning -y -ss $c.start -i $input -t $c.dur -c copy $c.out
}

Write-Host "Done. Created 12 chunks with 3-second overlap."
