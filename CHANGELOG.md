# Changelog

## [0.2.6](https://github.com/beamform/pacmap-rs/compare/v0.2.5...v0.2.6) (2024-12-02)


### Performance Improvements

* increase chunk size and reduce allocation ([#24](https://github.com/beamform/pacmap-rs/issues/24)) ([4d1f2c7](https://github.com/beamform/pacmap-rs/commit/4d1f2c750366ff265b446d389d12edce644131f8))

## [0.2.5](https://github.com/beamform/pacmap-rs/compare/v0.2.4...v0.2.5) (2024-11-18)


### Bug Fixes

* skip zero pairs ([#21](https://github.com/beamform/pacmap-rs/issues/21)) ([266957c](https://github.com/beamform/pacmap-rs/commit/266957c8ece2ab2a03aa91e224fea6c26720b490))

## [0.2.4](https://github.com/beamform/pacmap-rs/compare/v0.2.3...v0.2.4) (2024-11-17)


### Bug Fixes

* eliminate panic when zip lengths don’t match ([#19](https://github.com/beamform/pacmap-rs/issues/19)) ([a205233](https://github.com/beamform/pacmap-rs/commit/a205233e96d395f38f74c69f380d3b98dd154657))

## [0.2.3](https://github.com/beamform/pacmap-rs/compare/v0.2.2...v0.2.3) (2024-11-16)


### Bug Fixes

* rejection sampling in sample_FP and sample_MN ([#17](https://github.com/beamform/pacmap-rs/issues/17)) ([b94a785](https://github.com/beamform/pacmap-rs/commit/b94a7851d75dd6a3b513abd7ad3779e6bdda0526))

## [0.2.2](https://github.com/beamform/pacmap-rs/compare/v0.2.1...v0.2.2) (2024-11-13)


### Bug Fixes

* make simd a feature flag ([#15](https://github.com/beamform/pacmap-rs/issues/15)) ([7783939](https://github.com/beamform/pacmap-rs/commit/7783939af19c625b3f9e6b0b1f619c78477df213))

## [0.2.1](https://github.com/beamform/pacmap-rs/compare/v0.2.0...v0.2.1) (2024-11-13)


### Bug Fixes

* downgrade usearch to compile on older OSes ([#12](https://github.com/beamform/pacmap-rs/issues/12)) ([dde9884](https://github.com/beamform/pacmap-rs/commit/dde9884eab427a9adf668bdf50d08ba96dc3792c))

## [0.2.0](https://github.com/beamform/pacmap-rs/compare/v0.1.0...v0.2.0) (2024-11-13)


### Features

* **knn:** add approximate nearest neighbor search with USearch ([#8](https://github.com/beamform/pacmap-rs/issues/8)) ([3c2bb44](https://github.com/beamform/pacmap-rs/commit/3c2bb440d312d5fdb35ee3e3f5f660eab7542aa1))

## 0.1.0 (2024-11-05)

### Features

* PaCMAP initial
  commit ([45e2902](https://github.com/beamform/pacmap-rs/commit/45e290235bb5bac72bbb6b4483ec1d9eeadb46df))

### Bug Fixes

* handle low row
  counts ([f05aee6](https://github.com/beamform/pacmap-rs/commit/f05aee613da57c182342a12c415b5fc44c4e9514))
* repository URL ([2526720](https://github.com/beamform/pacmap-rs/commit/25267200067a124eadd1a8b27b28a5fc3da07391))

### Miscellaneous Chores

* release 0.1.0 ([a7651ac](https://github.com/beamform/pacmap-rs/commit/a7651ac079d65a2630215798f9178f33d54077c6))
