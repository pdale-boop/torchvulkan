#pragma once
#include <ATen/ATen.h>
#include <map>
#include <optional>

#include "builders.h"
#include "shader.h"

struct DeviceContext;

struct TuneConfig
{
    std::vector<uint32_t> values;
    uint32_t operator[](size_t i) const { return values[i]; }
    bool empty() const { return values.empty(); }
};

struct TuneParam
{
    std::string name;
    std::vector<uint32_t> options;
    size_t size() const { return options.size(); }
};

struct TuneSpace
{
    std::vector<TuneParam> params;
    std::function<bool(const TuneConfig&)> valid;
    std::optional<size_t> kernel_param;
    std::vector<TuneConfig> configs() const;
};

struct TuneDispatch
{
    ShaderKey key{};
    uint32_t subgroupSize = 0;
    SpecializationBuilder spd;
    PushConstantBuilder pcs;
    std::vector<at::Tensor> read;
    std::vector<at::Tensor> write;
    uint32_t groupX = 1, groupY = 1, groupZ = 1;
    double extra_ns = 0; // work the config adds outside this dispatch, added to its time
};

class Autotuner
{
public:
    explicit Autotuner(DeviceContext* device);

    // check, when given, is called after the chosen config has run once and says whether its output is right; a config
    // that fails is left out and the search runs again, so a miscompiled config is never saved
    const TuneConfig& select(
        ShaderKey shader,
        uint64_t tuned_id,
        const TuneSpace& space,
        const std::function<TuneDispatch(const TuneConfig&)>& build,
        const std::function<bool()>& check = {}
    );

private:
    DeviceContext* device = nullptr;
    std::mutex mutex_;
    std::map<std::pair<uint32_t, uint64_t>, TuneConfig> results_; // (shader binary, tuned_id)
    std::map<std::pair<uint64_t, uint64_t>, TuneConfig> stored_;  // (binary hash, tuned_id)
    std::string path_;
    bool loaded_ = false;

    double budget_seconds_; // a search settles for its best so far after this long; 0 for no limit
    bool log_;
    bool check_;            // TORCHVULKAN_TUNE_CHECK=0 turns the output checks off
    size_t threads_;        // compiling pipelines; GPU timing stays serial

    void load();
    void save(uint64_t hash, uint64_t tuned_id, const TuneConfig& config);

    TuneConfig fastest(const TuneSpace& space, const std::function<TuneDispatch(const TuneConfig&)>& build);
    TuneConfig checked(
        const TuneSpace& space,
        const std::function<TuneDispatch(const TuneConfig&)>& build,
        const std::function<bool()>& check,
        TuneConfig best
    );
    void compile(const std::vector<TuneConfig>& configs, const std::function<TuneDispatch(const TuneConfig&)>& build);
};
