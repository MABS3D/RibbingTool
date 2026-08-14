def test_fixtures_exist(box_step, cyl_patch_step, sphere_patch_step, full_cyl_step,
                        part1_path, part2_path):
    for p in (box_step, cyl_patch_step, sphere_patch_step, full_cyl_step,
              part1_path, part2_path):
        assert p.exists() and p.stat().st_size > 0
